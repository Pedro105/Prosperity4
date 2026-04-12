#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datamodel import Listing, Observation, Order, OrderDepth, TradingState

# ─────────────────────────────────────────────
# STRATEGY PARAMETERS — tune these
# ─────────────────────────────────────────────

# TOMATOES — Mean Reversion
EMA_WINDOW = 5
REVERSION_BETA = -0.42
TAKE_WIDTH = 1.5
CLEAR_WIDTH = 0
DEFAULT_EDGE = 1
JOIN_EDGE = 0
DISREGARD_EDGE = 1
PREVENT_ADVERSE = True
ADVERSE_VOLUME = 15
EMA_ALPHA = 2.0 / (EMA_WINDOW + 1)

# EMERALDS — copied from trader_linreg.py
PARAMS = {
    "EMERALDS": {
        "fair_value": 10000,
        "take_edge": 1,
        "flatten_threshold": 45,
        "spread_half": 1,
        "inventory_skew": 0.10,
        "base_size": 28,
        "min_size": 8,
        "size_position_step": 4,
    }
}

LIMITS = {"EMERALDS": 80, "TOMATOES": 80}
WORKSPACE = Path(__file__).resolve().parent
REPO_ROOT = WORKSPACE.parent
RUST_BACKTESTER = REPO_ROOT / "prosperity_rust_backtester-main"
RESULTS_DIR = WORKSPACE / "results"
RUST_RUNS_DIR = RESULTS_DIR / "rust_runs"
PLOTS_DIR = REPO_ROOT / "plots" / "trader_mr"


class Trader:
    def run(self, state: TradingState):
        memory = {}
        if state.traderData:
            try:
                memory = jsonpickle.decode(state.traderData)
            except Exception:
                memory = {}

        result: Dict[str, List[Order]] = {}
        for product, depth in state.order_depths.items():
            pos = state.position.get(product, 0)
            if product == "EMERALDS":
                result[product] = self.trade_emeralds(depth, pos)
            elif product == "TOMATOES":
                result[product] = self.trade_tomatoes(copy.deepcopy(depth), pos, memory)
        return result, 0, jsonpickle.encode(memory)

    def best_bid_ask(self, depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        bid = max(depth.buy_orders) if depth.buy_orders else None
        ask = min(depth.sell_orders) if depth.sell_orders else None
        return bid, ask

    def clamp(self, value: int, low: int, high: int) -> int:
        return max(low, min(high, value))

    # EMERALDS — unchanged from trader_linreg.py
    def take_liquidity_buy(self, product: str, orders: List[Order], depth: OrderDepth, max_price: int, position: int, limit: int) -> int:
        for ask in sorted(depth.sell_orders):
            if ask > max_price:
                break
            qty = min(-depth.sell_orders[ask], limit - position)
            if qty > 0:
                orders.append(Order(product, ask, qty))
                position += qty
        return position

    def take_liquidity_sell(self, product: str, orders: List[Order], depth: OrderDepth, min_price: int, position: int, limit: int) -> int:
        for bid in sorted(depth.buy_orders, reverse=True):
            if bid < min_price:
                break
            qty = min(depth.buy_orders[bid], limit + position)
            if qty > 0:
                orders.append(Order(product, bid, -qty))
                position -= qty
        return position

    def trade_emeralds(self, depth: OrderDepth, position: int) -> List[Order]:
        product = "EMERALDS"
        cfg = PARAMS[product]
        limit = LIMITS[product]
        fair = cfg["fair_value"]
        orders: List[Order] = []
        best_bid, best_ask = self.best_bid_ask(depth)

        position = self.take_liquidity_buy(product, orders, depth, fair - cfg["take_edge"], position, limit)
        position = self.take_liquidity_sell(product, orders, depth, fair + cfg["take_edge"], position, limit)
        if position > cfg["flatten_threshold"]:
            position = self.take_liquidity_sell(product, orders, depth, fair, position, limit)
        elif position < -cfg["flatten_threshold"]:
            position = self.take_liquidity_buy(product, orders, depth, fair, position, limit)

        reservation = fair - cfg["inventory_skew"] * position
        bid_quote = math.floor(reservation - cfg["spread_half"])
        ask_quote = math.ceil(reservation + cfg["spread_half"])
        if best_bid is not None:
            bid_quote = min(bid_quote, best_bid + 1)
        if best_ask is not None:
            ask_quote = max(ask_quote, best_ask - 1)
        if bid_quote >= ask_quote:
            bid_quote = ask_quote - 1
        bid_quote = min(bid_quote, fair)
        ask_quote = max(ask_quote, fair)

        buy_cap, sell_cap = limit - position, limit + position
        bid_size = self.clamp(cfg["base_size"] - max(position, 0) // cfg["size_position_step"], cfg["min_size"], buy_cap)
        ask_size = self.clamp(cfg["base_size"] + min(position, 0) // cfg["size_position_step"], cfg["min_size"], sell_cap)
        if buy_cap > 0 and bid_size > 0:
            orders.append(Order(product, bid_quote, bid_size))
        if sell_cap > 0 and ask_size > 0:
            orders.append(Order(product, ask_quote, -ask_size))
        return orders

    # TOMATOES — starfruit-style 3-phase pipeline
    def tomatoes_fair_value(self, depth: OrderDepth, memory: Dict[str, Any]) -> Optional[float]:
        prev_ema = memory.get("tomatoes_ema")
        best_bid, best_ask = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            if prev_ema is None:
                return None
            mmmid = float(prev_ema)
        else:
            filtered_asks = [p for p, v in depth.sell_orders.items() if abs(v) >= ADVERSE_VOLUME]
            filtered_bids = [p for p, v in depth.buy_orders.items() if abs(v) >= ADVERSE_VOLUME]
            mm_ask = min(filtered_asks) if filtered_asks else None
            mm_bid = max(filtered_bids) if filtered_bids else None
            if mm_ask is None or mm_bid is None:
                mmmid = float(memory.get("tomatoes_last_price", (best_bid + best_ask) / 2.0))
            else:
                mmmid = (mm_ask + mm_bid) / 2.0

        ema = mmmid if prev_ema is None else EMA_ALPHA * mmmid + (1.0 - EMA_ALPHA) * float(prev_ema)
        last_price = memory.get("tomatoes_last_price")
        fair = mmmid
        if last_price is not None:
            last_return = (mmmid - float(last_price)) / float(last_price)
            fair = mmmid + mmmid * (last_return * REVERSION_BETA)

        memory["tomatoes_ema"] = ema
        memory["tomatoes_last_price"] = mmmid
        memory["tomatoes_last_fair"] = fair
        return fair

    def take_orders(self, product: str, depth: OrderDepth, fair: float, take_width: float, position: int, prevent_adverse: bool = False, adverse_volume: int = 0) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_v = sell_v = 0
        limit = LIMITS[product]
        if depth.sell_orders:
            ask = min(depth.sell_orders)
            ask_vol = -depth.sell_orders[ask]
            if (not prevent_adverse or abs(ask_vol) <= adverse_volume) and ask <= fair - take_width:
                qty = min(ask_vol, limit - position)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    buy_v += qty
                    depth.sell_orders[ask] += qty
                    if depth.sell_orders[ask] == 0:
                        del depth.sell_orders[ask]
        if depth.buy_orders:
            bid = max(depth.buy_orders)
            bid_vol = depth.buy_orders[bid]
            if (not prevent_adverse or abs(bid_vol) <= adverse_volume) and bid >= fair + take_width:
                qty = min(bid_vol, limit + position)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_v += qty
                    depth.buy_orders[bid] -= qty
                    if depth.buy_orders[bid] == 0:
                        del depth.buy_orders[bid]
        return orders, buy_v, sell_v

    def clear_orders(self, product: str, depth: OrderDepth, fair: float, clear_width: int, position: int, buy_v: int, sell_v: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        pos_after_take = position + buy_v - sell_v
        bid_px = math.floor(fair - clear_width)
        ask_px = math.ceil(fair + clear_width)
        buy_cap = LIMITS[product] - (position + buy_v)
        sell_cap = LIMITS[product] + (position - sell_v)
        if pos_after_take > 0:
            clear_qty = min(sum(v for p, v in depth.buy_orders.items() if p >= ask_px), pos_after_take)
            sent = min(sell_cap, clear_qty)
            if sent > 0:
                orders.append(Order(product, ask_px, -sent))
                sell_v += sent
        if pos_after_take < 0:
            clear_qty = min(sum(abs(v) for p, v in depth.sell_orders.items() if p <= bid_px), abs(pos_after_take))
            sent = min(buy_cap, clear_qty)
            if sent > 0:
                orders.append(Order(product, bid_px, sent))
                buy_v += sent
        return orders, buy_v, sell_v

    def make_orders(self, product: str, depth: OrderDepth, fair: float, position: int, buy_v: int, sell_v: int, disregard_edge: float, join_edge: float, default_edge: float) -> List[Order]:
        asks_above = [p for p in depth.sell_orders if p > fair + disregard_edge]
        bids_below = [p for p in depth.buy_orders if p < fair - disregard_edge]
        best_ask = min(asks_above) if asks_above else None
        best_bid = max(bids_below) if bids_below else None
        ask = math.ceil(fair + default_edge) if best_ask is None else (best_ask if abs(best_ask - fair) <= join_edge else best_ask - 1)
        bid = math.floor(fair - default_edge) if best_bid is None else (best_bid if abs(fair - best_bid) <= join_edge else best_bid + 1)
        orders: List[Order] = []
        buy_qty = LIMITS[product] - (position + buy_v)
        sell_qty = LIMITS[product] + (position - sell_v)
        if buy_qty > 0:
            orders.append(Order(product, int(math.floor(bid)), buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, int(math.ceil(ask)), -sell_qty))
        return orders

    def trade_tomatoes(self, depth: OrderDepth, position: int, memory: Dict[str, Any]) -> List[Order]:
        fair = self.tomatoes_fair_value(depth, memory)
        if fair is None:
            return []
        take_orders, buy_v, sell_v = self.take_orders("TOMATOES", depth, fair, TAKE_WIDTH, position, PREVENT_ADVERSE, ADVERSE_VOLUME)
        clear_orders, buy_v, sell_v = self.clear_orders("TOMATOES", depth, fair, CLEAR_WIDTH, position, buy_v, sell_v)
        make_orders = self.make_orders("TOMATOES", depth, fair, position, buy_v, sell_v, DISREGARD_EDGE, JOIN_EDGE, DEFAULT_EDGE)
        return take_orders + clear_orders + make_orders


def build_depth(row: pd.Series) -> OrderDepth:
    depth = OrderDepth()
    for level in range(1, 4):
        bp, bv = row.get(f"bid_price_{level}"), row.get(f"bid_volume_{level}")
        ap, av = row.get(f"ask_price_{level}"), row.get(f"ask_volume_{level}")
        if pd.notna(bp) and pd.notna(bv) and str(bp).strip():
            depth.buy_orders[int(bp)] = int(bv)
        if pd.notna(ap) and pd.notna(av) and str(ap).strip():
            depth.sell_orders[int(ap)] = -abs(int(av))
    return depth


def listings() -> Dict[str, Listing]:
    return {"EMERALDS": Listing("EMERALDS", "EMERALDS", "SEASHELLS"), "TOMATOES": Listing("TOMATOES", "TOMATOES", "SEASHELLS")}


def replay_for_plot(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";").sort_values("timestamp")
    trader, trader_data = Trader(), ""
    rows = []
    for ts, group in df.groupby("timestamp", sort=True):
        rows_by_product = {row["product"]: row for _, row in group.iterrows()}
        if "EMERALDS" not in rows_by_product or "TOMATOES" not in rows_by_product:
            continue
        state = TradingState(trader_data, int(ts), listings(), {"EMERALDS": build_depth(rows_by_product["EMERALDS"]), "TOMATOES": build_depth(rows_by_product["TOMATOES"])}, {"EMERALDS": [], "TOMATOES": []}, {"EMERALDS": [], "TOMATOES": []}, {"EMERALDS": 0, "TOMATOES": 0}, Observation({}, {}))
        result, _, trader_data = trader.run(state)
        memory = jsonpickle.decode(trader_data) if trader_data else {}
        row = rows_by_product["TOMATOES"]
        mid = float(row["mid_price"]) if pd.notna(row["mid_price"]) else np.nan
        tomatoes_orders = result.get("TOMATOES", [])
        if not tomatoes_orders:
            rows.append({"timestamp": int(ts), "mid": mid, "ema": memory.get("tomatoes_ema"), "fair": memory.get("tomatoes_last_fair"), "price": np.nan, "side": None})
        for order in tomatoes_orders:
            rows.append({"timestamp": int(ts), "mid": mid, "ema": memory.get("tomatoes_ema"), "fair": memory.get("tomatoes_last_fair"), "price": order.price, "side": "buy" if order.quantity > 0 else "sell"})
    return pd.DataFrame(rows)


def max_drawdown(series: np.ndarray) -> float:
    return float(np.min(series - np.maximum.accumulate(series))) if len(series) else float("nan")


def sharpe(series: np.ndarray) -> float:
    if len(series) < 2:
        return float("nan")
    returns = np.diff(series)
    sigma = np.std(returns, ddof=1)
    return float(np.mean(returns) / sigma) if sigma else float("nan")


def run_tutorial_backtest(trader_path: Path) -> Optional[Path]:
    if not RUST_BACKTESTER.is_dir():
        return None
    run_id = f"trader-mr-{int(time.time() * 1000)}"
    output_root = RUST_RUNS_DIR
    run_results_dir = RESULTS_DIR / trader_path.stem / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    run_results_dir.mkdir(parents=True, exist_ok=True)
    python_executable = Path(sys.executable).resolve()
    python_libdir = Path(sysconfig.get_config_var("LIBDIR")).resolve()
    cargo_target_dir = output_root / "target-py" / f"{sys.version_info.major}.{sys.version_info.minor}"
    command = (
        f'source "$HOME/.cargo/env" && '
        f'PYO3_PYTHON="{python_executable}" '
        f'DYLD_FALLBACK_LIBRARY_PATH="{python_libdir}:${{DYLD_FALLBACK_LIBRARY_PATH:-}}" '
        f'CARGO_TARGET_DIR="{cargo_target_dir}" '
        f'./scripts/cargo_local.sh run -- '
        f'--trader "{trader_path}" --dataset "tutorial" --artifact-mode diagnostic --flat '
        f'--run-id "{run_id}" --output-root "{output_root}"'
    )
    completed = subprocess.run(["bash", "-lc", command], cwd=RUST_BACKTESTER, text=True)
    if completed.returncode != 0:
        return None
    run_dir = output_root / run_id
    if not run_dir.exists():
        return None
    summary_md = run_results_dir / f"{trader_path.stem}-summary.md"
    summary_csv = run_results_dir / f"{trader_path.stem}-summary.csv"
    rows = []
    for metrics_path in sorted(run_dir.glob("*-metrics.json")):
        with metrics_path.open() as f:
            rows.append(json.load(f))
    summary_md.write_text("\n".join(["| Dataset | Day | Total PnL | EMERALDS | TOMATOES | Own Trades | Ticks |", "|---|---:|---:|---:|---:|---:|---:|"] + [f"| {r.get('dataset_id','')} | {r.get('day','')} | {r.get('final_pnl_total',0.0):.2f} | {(r.get('final_pnl_by_product') or {}).get('EMERALDS',0.0):.2f} | {(r.get('final_pnl_by_product') or {}).get('TOMATOES',0.0):.2f} | {r.get('own_trade_count',0)} | {r.get('tick_count',0)} |" for r in rows]) + "\n")
    summary_csv.write_text("\n".join(["dataset_id,day,total_pnl,emeralds_pnl,tomatoes_pnl,own_trade_count,tick_count"] + [f"{r.get('dataset_id','')},{r.get('day','')},{r.get('final_pnl_total',0.0):.2f},{(r.get('final_pnl_by_product') or {}).get('EMERALDS',0.0):.2f},{(r.get('final_pnl_by_product') or {}).get('TOMATOES',0.0):.2f},{r.get('own_trade_count',0)},{r.get('tick_count',0)}" for r in rows]) + "\n")
    png_paths = []
    for bundle_path in sorted(run_dir.glob("*-bundle.json")):
        with bundle_path.open() as f:
            bundle = json.load(f)
        pnl_series = bundle.get("pnl_series", [])
        if not pnl_series:
            continue
        timestamps = [row["timestamp"] for row in pnl_series]
        totals = [row["total"] for row in pnl_series]
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = {"EMERALDS": "#2ecc71", "TOMATOES": "#e74c3c", "TOTAL": "#3498db"}
        for product in ["EMERALDS", "TOMATOES"]:
            vals = [row.get("by_product", {}).get(product, 0.0) for row in pnl_series]
            ax.plot(timestamps, vals, label=product, linewidth=1.2, alpha=0.9, color=colors[product])
        ax.plot(timestamps, totals, label="TOTAL", linewidth=1.8, linestyle="--", alpha=0.9, color=colors["TOTAL"])
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.set_title(bundle_path.stem.replace("-bundle", "").replace("-", " "))
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("PnL")
        ax.legend()
        fig.tight_layout()
        png_path = run_results_dir / f"{trader_path.stem}-{bundle_path.stem.replace('-bundle', '')}.png"
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        png_paths.append(png_path)
    if png_paths:
        latest_png = next((p for p in png_paths if "submission" in p.name), png_paths[-1])
        latest_trader_png = RESULTS_DIR / trader_path.stem / f"latest-{trader_path.stem}.png"
        latest_summary_md = RESULTS_DIR / trader_path.stem / f"latest-{trader_path.stem}-summary.md"
        latest_summary_csv = RESULTS_DIR / trader_path.stem / f"latest-{trader_path.stem}-summary.csv"
        latest_trader_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(latest_png, latest_trader_png)
        shutil.copyfile(summary_md, latest_summary_md)
        shutil.copyfile(summary_csv, latest_summary_csv)
        shutil.copyfile(latest_png, WORKSPACE / "pnl_curve.png")
    return run_dir


def print_pnl_report(run_dir: Path) -> Dict[Optional[int], Dict[str, Any]]:
    out: Dict[Optional[int], Dict[str, Any]] = {}
    metrics_paths = sorted(run_dir.glob("*-metrics.json"))
    print("\n" + "=" * 80)
    print("TUTORIAL BACKTEST — Final PnL")
    print("=" * 80)
    print(f"{'Day':<10}{'EMERALDS':>14}{'TOMATOES':>14}{'TOTAL':>14}{'Trades':>14}{'Ticks':>14}")
    print("-" * 80)
    for path in metrics_paths:
        with path.open() as f:
            metrics = json.load(f)
        day = metrics.get("day")
        by_product = metrics.get("final_pnl_by_product", {})
        out[day] = {
            "emeralds": float(by_product.get("EMERALDS", 0.0)),
            "tomatoes": float(by_product.get("TOMATOES", 0.0)),
            "total": float(metrics.get("final_pnl_total", 0.0)),
            "trades": int(metrics.get("own_trade_count", 0)),
            "ticks": int(metrics.get("tick_count", 0)),
        }
        print(f"{str(day):<10}{out[day]['emeralds']:14.2f}{out[day]['tomatoes']:14.2f}{out[day]['total']:14.2f}{out[day]['trades']:14d}{out[day]['ticks']:14d}")
    print("=" * 80)
    return out


def load_bundle_stats(run_dir: Path) -> Dict[Optional[int], Dict[str, float]]:
    out: Dict[Optional[int], Dict[str, float]] = {}
    for path in sorted(run_dir.glob("*-bundle.json")):
        with path.open() as f:
            bundle = json.load(f)
        day = bundle.get("run", {}).get("day")
        totals = np.array([row.get("total", 0.0) for row in bundle.get("pnl_series", [])], dtype=float)
        out[day] = {"max_drawdown": max_drawdown(totals), "sharpe": sharpe(totals)}
    return out


def plot_day(day_label: str, csv_path: Path, perf: Optional[Dict[str, Any]], risk: Optional[Dict[str, float]]) -> None:
    data = replay_for_plot(csv_path)
    if data.empty:
        return
    grouped = data.groupby("timestamp", as_index=False).first()
    buys = data[data["side"] == "buy"]
    sells = data[data["side"] == "sell"]
    out_path = PLOTS_DIR / f"tomatoes_{csv_path.stem.replace('prices_round_0_', '').replace('day_', 'day')}.png"
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(grouped["timestamp"], grouped["mid"], color="lightgrey", linewidth=0.8, label="Mid price")
    ax.plot(grouped["timestamp"], grouped["ema"], color="steelblue", linewidth=1.2, label="EMA")
    ax.plot(grouped["timestamp"], grouped["fair"], color="darkorange", linewidth=1.2, linestyle="--", label="Beta-adjusted fair")
    ax.scatter(buys["timestamp"], buys["price"], color="green", marker="^", s=28, label="BUY orders")
    ax.scatter(sells["timestamp"], sells["price"], color="red", marker="v", s=28, label="SELL orders")
    ax.set_title(f"{day_label} — TOMATOES — EMA({EMA_WINDOW}), β={REVERSION_BETA}")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("price")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")
    print(f"  Number of buys: {len(buys)}")
    print(f"  Number of sells: {len(sells)}")
    print("  Final position: 0")
    if perf is not None:
        print(f"  EMERALDS PnL: {perf['emeralds']:.2f}")
        print(f"  TOMATOES PnL: {perf['tomatoes']:.2f}")
        print(f"  Total PnL: {perf['total']:.2f}")
    if risk is not None:
        print(f"  Max drawdown: {risk['max_drawdown']:.4f}")
        print(f"  Sharpe ratio: {risk['sharpe']:.4f}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n--- Running tutorial backtest ---")
    run_dir = run_tutorial_backtest(Path(__file__).resolve())
    perf_by_day = print_pnl_report(run_dir) if run_dir else {}
    risk_by_day = load_bundle_stats(run_dir) if run_dir else {}

    for day, path in [(-2, WORKSPACE / "prices_round_0_day_-2.csv"), (-1, WORKSPACE / "prices_round_0_day_-1.csv"), (0, WORKSPACE / "prices_round_0_day_0.csv")]:
        if not path.is_file():
            print(f"Skip missing dataset: {path}")
            continue
        plot_day(f"Day {day}", path, perf_by_day.get(day), risk_by_day.get(day))


if __name__ == "__main__":
    main()

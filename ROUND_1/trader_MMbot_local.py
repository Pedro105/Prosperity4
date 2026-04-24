#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "TUTORIAL_ROUND_1"))

from datamodel import Listing, Observation, OrderDepth, TradingState
from trader_MMbot import OSMIUM, PEPPER, Trader

WORKSPACE = Path(__file__).resolve().parent
REPO_ROOT = WORKSPACE.parent
RUST_BACKTESTER = REPO_ROOT / "prosperity_rust_backtester-main"
RESULTS_DIR = WORKSPACE / "results"
RUST_RUNS_DIR = RESULTS_DIR / "rust_runs"
PLOTS_DIR = WORKSPACE / "plots" / "trader_MMbot"
TRADER_PATH = WORKSPACE / "trader_MMbot.py"


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
    return {PEPPER: Listing(PEPPER, PEPPER, "SEASHELLS"), OSMIUM: Listing(OSMIUM, OSMIUM, "SEASHELLS")}


def replay_for_plot(csv_path: Path, product: str, fair_key: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";").sort_values("timestamp")
    trader, trader_data = Trader(), ""
    rows = []
    for ts, group in df.groupby("timestamp", sort=True):
        rows_by_product = {row["product"]: row for _, row in group.iterrows()}
        if PEPPER not in rows_by_product or OSMIUM not in rows_by_product:
            continue
        state = TradingState(
            trader_data,
            int(ts),
            listings(),
            {PEPPER: build_depth(rows_by_product[PEPPER]), OSMIUM: build_depth(rows_by_product[OSMIUM])},
            {PEPPER: [], OSMIUM: []},
            {PEPPER: [], OSMIUM: []},
            {PEPPER: 0, OSMIUM: 0},
            Observation({}, {}),
        )
        result, _, trader_data = trader.run(state)
        memory = json.loads(trader_data) if trader_data else {}
        row = rows_by_product[product]
        mid = float(row["mid_price"]) if pd.notna(row["mid_price"]) else np.nan
        product_orders = result.get(product, [])
        if not product_orders:
            rows.append({"timestamp": int(ts), "mid": mid, "fair": memory.get(fair_key), "price": np.nan, "side": None})
        for order in product_orders:
            rows.append({"timestamp": int(ts), "mid": mid, "fair": memory.get(fair_key), "price": order.price, "side": "buy" if order.quantity > 0 else "sell"})
    return pd.DataFrame(rows)


def plot_signal(day_label: str, csv_path: Path, product: str, fair_key: str) -> None:
    data = replay_for_plot(csv_path, product, fair_key)
    if data.empty:
        return
    grouped = data.groupby("timestamp", as_index=False).first()
    buys = data[data["side"] == "buy"]
    sells = data[data["side"] == "sell"]
    out_path = PLOTS_DIR / f"{product.lower()}_{csv_path.stem}.png"
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(grouped["timestamp"], grouped["mid"], color="lightgrey", linewidth=0.8, label="Mid price")
    ax.plot(grouped["timestamp"], grouped["fair"], color="steelblue", linewidth=1.2, linestyle="--", label="Fair value")
    ax.scatter(buys["timestamp"], buys["price"], color="green", marker="^", s=28, label="BUY orders")
    ax.scatter(sells["timestamp"], sells["price"], color="red", marker="v", s=28, label="SELL orders")
    ax.set_title(f"{day_label} — {product}")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("price")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")


def max_drawdown(series: np.ndarray) -> tuple[float, int]:
    if len(series) == 0:
        return float("nan"), -1
    running_max = np.maximum.accumulate(series)
    drawdowns = series - running_max
    trough_idx = int(np.argmin(drawdowns))
    return float(drawdowns[trough_idx]), trough_idx


def recovery_time(timestamps: list[int], totals: np.ndarray, trough_idx: int) -> Optional[int]:
    if trough_idx < 0 or len(totals) == 0:
        return None
    peak_before = float(np.max(totals[: trough_idx + 1]))
    trough_ts = timestamps[trough_idx]
    for i in range(trough_idx + 1, len(totals)):
        if totals[i] >= peak_before:
            return int(timestamps[i] - trough_ts)
    return None


def average_fill_sizes(timeline: list[dict]) -> Dict[str, float]:
    fills: Dict[str, list[int]] = {PEPPER: [], OSMIUM: []}
    for row in timeline:
        for trade in row.get("own_trades", []):
            symbol = trade.get("symbol")
            qty = abs(int(trade.get("quantity", 0)))
            if symbol in fills and qty > 0:
                fills[symbol].append(qty)
    result = {}
    for symbol, values in fills.items():
        result[symbol] = float(np.mean(values)) if values else 0.0
    all_values = fills[PEPPER] + fills[OSMIUM]
    result["ALL"] = float(np.mean(all_values)) if all_values else 0.0
    return result


def run_round1_backtest(trader_path: Path) -> Optional[Path]:
    if not RUST_BACKTESTER.is_dir():
        return None
    run_id = f"mmbot-{int(time.time() * 1000)}"
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
        f'--trader "{trader_path}" --dataset "{WORKSPACE}" --artifact-mode full --flat '
        f'--run-id "{run_id}" --output-root "{output_root}"'
    )
    completed = subprocess.run(["bash", "-lc", command], cwd=RUST_BACKTESTER, text=True)
    if completed.returncode != 0:
        return None
    run_dir = output_root / run_id
    if not run_dir.exists():
        return None

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
        for product, color in [(PEPPER, "#2ecc71"), (OSMIUM, "#e74c3c")]:
            vals = [row.get("by_product", {}).get(product, 0.0) for row in pnl_series]
            ax.plot(timestamps, vals, label=product, linewidth=1.2, alpha=0.9, color=color)
        ax.plot(timestamps, totals, label="TOTAL", linewidth=1.8, linestyle="--", alpha=0.9, color="#3498db")
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
        print(f"Generated {png_path}")

    if png_paths:
        latest_png = png_paths[-1]
        latest_trader_png = RESULTS_DIR / trader_path.stem / f"latest-{trader_path.stem}.png"
        latest_trader_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(latest_png, latest_trader_png)
    return run_dir


def print_pnl_report(run_dir: Path) -> None:
    metric_rows = []
    for path in run_dir.glob("*-metrics.json"):
        with path.open() as f:
            metrics = json.load(f)
        metric_rows.append(metrics)
    print("\n" + "=" * 88)
    print("ROUND 1 BACKTEST — Final PnL")
    print("=" * 88)
    print(f"{'Day':<10}{PEPPER:>24}{OSMIUM:>24}{'TOTAL':>14}")
    print("-" * 88)
    for metrics in sorted(metric_rows, key=lambda row: int(row.get("day", 0))):
        day = metrics.get("day")
        by_product = metrics.get("final_pnl_by_product", {})
        pepper = float(by_product.get(PEPPER, 0.0))
        osmium = float(by_product.get(OSMIUM, 0.0))
        total = float(metrics.get("final_pnl_total", 0.0))
        print(f"{str(day):<10}{pepper:24.2f}{osmium:24.2f}{total:14.2f}")
    print("=" * 88)


def print_day_diagnostics(run_dir: Path) -> None:
    bundles = []
    for path in run_dir.glob("*-bundle.json"):
        with path.open() as f:
            bundles.append(json.load(f))
    print("\n" + "=" * 88)
    print("Diagnostics")
    print("=" * 88)
    for bundle in sorted(bundles, key=lambda row: int(row.get("run", {}).get("day", 0))):
        day = bundle.get("run", {}).get("day")
        pnl_series = bundle.get("pnl_series", [])
        timeline = bundle.get("timeline", [])
        timestamps = [int(row["timestamp"]) for row in pnl_series]
        totals = np.array([row.get("total", 0.0) for row in pnl_series], dtype=float)
        dd, trough_idx = max_drawdown(totals)
        recovery = recovery_time(timestamps, totals, trough_idx)
        fills = average_fill_sizes(timeline)
        print(f"Day {day}")
        print(f"  Max drawdown: {dd:.2f}")
        print(f"  Recovery: {'not recovered' if recovery is None else f'{recovery} timestamps'}")
        print(f"  Avg fill size ({PEPPER}): {fills[PEPPER]:.2f}")
        print(f"  Avg fill size ({OSMIUM}): {fills[OSMIUM]:.2f}")
        print(f"  Avg fill size (all): {fills['ALL']:.2f}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n--- Running Round 1 backtest for trader_MMbot ---")
    run_dir = run_round1_backtest(TRADER_PATH)
    if run_dir:
        print_pnl_report(run_dir)
        print_day_diagnostics(run_dir)
    else:
        print("Rust backtest did not complete; generating local signal plots only.")

    for day, path in [(-2, WORKSPACE / "prices_round_1_day_-2.csv"), (-1, WORKSPACE / "prices_round_1_day_-1.csv"), (0, WORKSPACE / "prices_round_1_day_0.csv")]:
        if not path.is_file():
            continue
        plot_signal(f"Day {day}", path, PEPPER, "pepper_fair")
        plot_signal(f"Day {day}", path, OSMIUM, "osmium_fair")


if __name__ == "__main__":
    main()

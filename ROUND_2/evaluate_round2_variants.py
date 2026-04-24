#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

import sys

sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "TUTORIAL_ROUND_1")))
from datamodel import Listing, Observation, Order, OrderDepth, Trade, TradingState


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "variant_comparison.txt"
PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
PRODUCTS = [PEPPER, OSMIUM]
DAYS = [-1, 0, 1]
VARIANTS = {
    "baseline": ROOT / "trader_round2_baseline.py",
    "current": ROOT / "trader_round2.py",
}


@dataclass
class VariantResult:
    name: str
    final_pnl_mid: float
    final_pnl_maker_mark: float
    trade_count: int
    own_volume: int
    final_positions: Dict[str, int]
    pnl_by_product_mid: Dict[str, float]
    pnl_by_product_maker: Dict[str, float]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def listings() -> Dict[str, Listing]:
    return {PEPPER: Listing(PEPPER, PEPPER, "SEASHELLS"), OSMIUM: Listing(OSMIUM, OSMIUM, "SEASHELLS")}


def build_depth(row: pd.Series) -> OrderDepth:
    depth = OrderDepth()
    for level in range(1, 4):
        bp, bv = row.get(f"bid_price_{level}"), row.get(f"bid_volume_{level}")
        ap, av = row.get(f"ask_price_{level}"), row.get(f"ask_volume_{level}")
        if pd.notna(bp) and pd.notna(bv):
            depth.buy_orders[int(bp)] = int(bv)
        if pd.notna(ap) and pd.notna(av):
            depth.sell_orders[int(ap)] = -abs(int(av))
    return depth


def load_day(day: int):
    prices = pd.read_csv(ROOT / f"prices_round_2_day_{day}.csv", sep=";")
    trades = pd.read_csv(ROOT / f"trades_round_2_day_{day}.csv", sep=";")
    prices = prices.sort_values(["timestamp", "product"]).reset_index(drop=True)
    trades = trades.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return prices, trades


def market_trades_for_timestamp(trades: pd.DataFrame, timestamp: int) -> Dict[str, List[Trade]]:
    snapshot = trades[trades["timestamp"] == timestamp]
    result = {PEPPER: [], OSMIUM: []}
    for row in snapshot.itertuples(index=False):
        symbol = row.symbol
        result[symbol].append(
            Trade(symbol, int(row.price), int(row.quantity), buyer=getattr(row, "buyer", None), seller=getattr(row, "seller", None), timestamp=int(row.timestamp))
        )
    return result


def fill_order(order: Order, row: pd.Series) -> List[Trade]:
    fills: List[Trade] = []
    remaining = abs(int(order.quantity))
    if remaining <= 0:
        return fills

    if order.quantity > 0:
        for level in range(1, 4):
            ask_p = row.get(f"ask_price_{level}")
            ask_q = row.get(f"ask_volume_{level}")
            if pd.isna(ask_p) or pd.isna(ask_q):
                continue
            ask_p = int(ask_p)
            ask_q = abs(int(ask_q))
            if ask_p > order.price:
                break
            traded = min(remaining, ask_q)
            if traded > 0:
                fills.append(Trade(order.symbol, ask_p, traded))
                remaining -= traded
            if remaining == 0:
                break
    else:
        for level in range(1, 4):
            bid_p = row.get(f"bid_price_{level}")
            bid_q = row.get(f"bid_volume_{level}")
            if pd.isna(bid_p) or pd.isna(bid_q):
                continue
            bid_p = int(bid_p)
            bid_q = int(bid_q)
            if bid_p < order.price:
                break
            traded = min(remaining, bid_q)
            if traded > 0:
                fills.append(Trade(order.symbol, bid_p, traded))
                remaining -= traded
            if remaining == 0:
                break
    return fills


def evaluate_variant(name: str, path: Path) -> VariantResult:
    module = load_module(path)
    trader = module.Trader()
    trader_data = ""
    position = {PEPPER: 0, OSMIUM: 0}
    cash = 0.0
    trade_count = 0
    own_volume = 0
    last_rows: Dict[str, pd.Series] = {}
    last_memory: Dict[str, float] = {}

    for day in DAYS:
        prices, trades = load_day(day)
        for timestamp, group in prices.groupby("timestamp", sort=True):
            rows = {row["product"]: row for _, row in group.iterrows()}
            last_rows = rows
            state = TradingState(
                trader_data,
                int(timestamp),
                listings(),
                {product: build_depth(rows[product]) for product in PRODUCTS},
                {PEPPER: [], OSMIUM: []},
                market_trades_for_timestamp(trades, int(timestamp)),
                dict(position),
                Observation({}, {}),
            )
            result, _, trader_data = trader.run(state)
            last_memory = json.loads(trader_data) if trader_data else {}

            own_trades = {PEPPER: [], OSMIUM: []}
            for product in PRODUCTS:
                for order in result.get(product, []):
                    fills = fill_order(order, rows[product])
                    for fill in fills:
                        signed_qty = fill.quantity if order.quantity > 0 else -fill.quantity
                        position[product] += signed_qty
                        cash -= signed_qty * fill.price
                        trade_count += 1
                        own_volume += fill.quantity
                        own_trades[product].append(fill)

    mark_mid = 0.0
    mark_maker = 0.0
    pnl_by_product_mid = {}
    pnl_by_product_maker = {}
    for product in PRODUCTS:
        row = last_rows[product]
        pos = position[product]
        mid = float(row["mid_price"])
        maker_mid = last_memory.get("osmium_maker_mid", mid) if product == OSMIUM else last_memory.get("pepper_fair", mid)
        product_mid = pos * mid
        product_maker = pos * maker_mid
        pnl_by_product_mid[product] = product_mid
        pnl_by_product_maker[product] = product_maker
        mark_mid += product_mid
        mark_maker += product_maker

    return VariantResult(
        name=name,
        final_pnl_mid=cash + mark_mid,
        final_pnl_maker_mark=cash + mark_maker,
        trade_count=trade_count,
        own_volume=own_volume,
        final_positions=dict(position),
        pnl_by_product_mid=pnl_by_product_mid,
        pnl_by_product_maker=pnl_by_product_maker,
    )


def main() -> None:
    results = [evaluate_variant(name, path) for name, path in VARIANTS.items()]
    lines = []
    lines.append("Round 2 variant comparison")
    lines.append("=" * 28)
    lines.append("")
    for result in results:
        lines.append(result.name)
        lines.append(f"  pnl_mid_mark: {result.final_pnl_mid:.2f}")
        lines.append(f"  pnl_maker_mark: {result.final_pnl_maker_mark:.2f}")
        lines.append(f"  trade_count: {result.trade_count}")
        lines.append(f"  own_volume: {result.own_volume}")
        lines.append(f"  final_positions: {result.final_positions}")
        lines.append(f"  pnl_by_product_mid: {result.pnl_by_product_mid}")
        lines.append(f"  pnl_by_product_maker: {result.pnl_by_product_maker}")
        lines.append("")

    if len(results) == 2:
        base, current = results
        lines.append("delta_current_minus_baseline")
        lines.append(f"  pnl_mid_mark: {current.final_pnl_mid - base.final_pnl_mid:.2f}")
        lines.append(f"  pnl_maker_mark: {current.final_pnl_maker_mark - base.final_pnl_maker_mark:.2f}")
        lines.append(f"  trade_count: {current.trade_count - base.trade_count}")
        lines.append(f"  own_volume: {current.own_volume - base.own_volume}")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

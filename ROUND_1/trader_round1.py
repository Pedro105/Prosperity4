#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from datamodel import Order, OrderDepth, TradingState
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "TUTORIAL_ROUND_1"))
    from datamodel import Order, OrderDepth, TradingState

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

# Derived from simple linear regression on day -2, -1, 0 mid-prices.
PEPPER_SLOPE = 0.0010000612960100698
PEPPER_HALF_SPREAD = 2
PEPPER_BUY_EDGE = 10
PEPPER_SELL_EDGE = 1
PEPPER_SIZE = 20
PEPPER_LATE_START = 990_000
PEPPER_FINAL_UNWIND = 999_000

OSMIUM_BETA = -0.4952
OSMIUM_EMA_ALPHA = 0.05
OSMIUM_HALF_SPREAD = 1
OSMIUM_TAKE_EDGE = 0
OSMIUM_INV_SKEW = 0.00
OSMIUM_SIZE = 14

LIMITS = {PEPPER: 80, OSMIUM: 80}


class Trader:
    def run(self, state: TradingState):
        memory = {}
        if state.traderData:
            try:
                memory = json.loads(state.traderData)
            except Exception:
                memory = {}

        result: Dict[str, List[Order]] = {}
        for product, depth in state.order_depths.items():
            pos = state.position.get(product, 0)
            if product == PEPPER:
                result[product] = self.trade_pepper(depth, pos, state.timestamp, memory)
            elif product == OSMIUM:
                result[product] = self.trade_osmium(depth, pos, memory)

        return result, 0, json.dumps(memory, separators=(",", ":"))

    def best_bid_ask(self, depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        bid = max(depth.buy_orders) if depth.buy_orders else None
        ask = min(depth.sell_orders) if depth.sell_orders else None
        return bid, ask

    def mid_price(self, depth: OrderDepth) -> Optional[float]:
        bid, ask = self.best_bid_ask(depth)
        return None if bid is None or ask is None else (bid + ask) / 2.0

    def quote_prices(self, fair: float, half_spread: float, inv_skew: float, position: int, depth: OrderDepth) -> Tuple[int, int]:
        reservation = fair - inv_skew * position
        raw_bid = math.floor(reservation - half_spread)
        raw_ask = math.ceil(reservation + half_spread)
        best_bid, best_ask = self.best_bid_ask(depth)
        bid = raw_bid if best_bid is None else min(raw_bid, best_bid + 1)
        ask = raw_ask if best_ask is None else max(raw_ask, best_ask - 1)
        if bid >= ask:
            bid = ask - 1
        return bid, ask

    def take_orders(self, product: str, fair: float, edge: float, depth: OrderDepth, position: int) -> Tuple[List[Order], int]:
        orders: List[Order] = []
        limit = LIMITS[product]
        net = 0
        if depth.sell_orders:
            ask = min(depth.sell_orders)
            ask_qty = -depth.sell_orders[ask]
            if ask <= fair - edge:
                qty = min(ask_qty, limit - position)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    net += qty
        if depth.buy_orders:
            bid = max(depth.buy_orders)
            bid_qty = depth.buy_orders[bid]
            if bid >= fair + edge:
                qty = min(bid_qty, limit + position - max(net, 0))
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    net -= qty
        return orders, net

    def make_orders(self, product: str, fair: float, depth: OrderDepth, position: int, net_after_take: int, half_spread: float, inv_skew: float) -> List[Order]:
        bid, ask = self.quote_prices(fair, half_spread, inv_skew, position + net_after_take, depth)
        buy_cap = LIMITS[product] - (position + max(net_after_take, 0))
        sell_cap = LIMITS[product] + (position + min(net_after_take, 0))
        orders: List[Order] = []
        if buy_cap > 0:
            orders.append(Order(product, bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, ask, -sell_cap))
        return orders

    def trade_pepper(self, depth: OrderDepth, position: int, timestamp: int, memory: Dict[str, float]) -> List[Order]:
        mid = self.mid_price(depth)
        if mid is None:
            return []

        if "pepper_t0_mid" not in memory:
            memory["pepper_t0_mid"] = float(mid)
            memory["pepper_t0_ts"] = int(timestamp)
        fair = memory["pepper_t0_mid"] + PEPPER_SLOPE * (timestamp - memory["pepper_t0_ts"])
        memory["pepper_fair"] = fair
        orders: List[Order] = []
        limit = LIMITS[PEPPER]
        best_bid, best_ask = self.best_bid_ask(depth)

        if timestamp < PEPPER_LATE_START:
            target_position = int(0.85 * limit)
        elif timestamp < PEPPER_FINAL_UNWIND:
            target_position = int(0.35 * limit)
        else:
            target_position = 0

        net = 0
        if best_ask is not None and position < target_position and best_ask <= fair + PEPPER_BUY_EDGE:
            qty = min(-depth.sell_orders[best_ask], target_position - position)
            if qty > 0:
                orders.append(Order(PEPPER, best_ask, qty))
                net += qty

        if best_bid is not None:
            desired_sell = max(0, position + net - target_position)
            if desired_sell > 0 and best_bid >= fair - PEPPER_SELL_EDGE:
                qty = min(depth.buy_orders[best_bid], desired_sell)
                if qty > 0:
                    orders.append(Order(PEPPER, best_bid, -qty))
                    net -= qty

        new_position = position + net
        buy_cap = limit - new_position
        sell_cap = limit + new_position

        if new_position < target_position and buy_cap > 0:
            bid_quote = math.floor(fair - PEPPER_HALF_SPREAD)
            if best_bid is not None:
                bid_quote = min(bid_quote, best_bid + 1)
            bid_qty = min(PEPPER_SIZE, target_position - new_position, buy_cap)
            if bid_qty > 0:
                orders.append(Order(PEPPER, bid_quote, bid_qty))

        if new_position > target_position and sell_cap > 0:
            ask_quote = math.ceil(fair + PEPPER_HALF_SPREAD)
            if best_ask is not None:
                ask_quote = max(ask_quote, best_ask - 1)
            ask_qty = min(PEPPER_SIZE, new_position - target_position, sell_cap)
            if ask_qty > 0:
                orders.append(Order(PEPPER, ask_quote, -ask_qty))

        return orders

    def trade_osmium(self, depth: OrderDepth, position: int, memory: Dict[str, float]) -> List[Order]:
        mid = self.mid_price(depth)
        if mid is None:
            last_fair = memory.get("osmium_fair")
            return [] if last_fair is None else self.make_orders(OSMIUM, last_fair, depth, position, 0, OSMIUM_HALF_SPREAD, OSMIUM_INV_SKEW)

        prev_mean = memory.get("osmium_mean")
        mean = mid if prev_mean is None else OSMIUM_EMA_ALPHA * mid + (1.0 - OSMIUM_EMA_ALPHA) * prev_mean
        predicted = mid + OSMIUM_BETA * (mid - mean)
        memory["osmium_mean"] = mean
        memory["osmium_fair"] = predicted

        take, net = self.take_orders(OSMIUM, predicted, OSMIUM_TAKE_EDGE, depth, position)
        make = self.make_orders(OSMIUM, predicted, depth, position, net, OSMIUM_HALF_SPREAD, OSMIUM_INV_SKEW)
        return take + make

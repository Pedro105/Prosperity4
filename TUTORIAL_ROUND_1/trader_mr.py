#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ─────────────────────────────────────────────
# STRATEGY PARAMETERS 
# ─────────────────────────────────────────────

# TOMATOES — Mean Reversion
EMA_WINDOW = 5
MR_SIGNAL_STRENGTH = 0.85
MR_ZSCORE_THRESHOLD = 0.75
MR_ZSCORE_CLIP = 3.0
AVG_MOVE_LOOKBACK = 8
MIN_AVG_MOVE = 1.5
TAKE_WIDTH = 1
CLEAR_WIDTH = 0
DEFAULT_EDGE = 1
JOIN_EDGE = 0
DISREGARD_EDGE = 1
PREVENT_ADVERSE = True
ADVERSE_VOLUME = 18
EMA_ALPHA = 2.0 / (EMA_WINDOW + 1)

# EMERALDS
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
            if product == "EMERALDS":
                result[product] = self.trade_emeralds(depth, pos)
            elif product == "TOMATOES":
                result[product] = self.trade_tomatoes(copy.deepcopy(depth), pos, memory)
        return result, 0, json.dumps(memory, separators=(",", ":"))

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

    # TOMATOES — 3-phase pipeline with normalized MR signal
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

        history = memory.setdefault("tomatoes_mid_history", [])
        history.append(mmmid)
        if len(history) > AVG_MOVE_LOOKBACK + 1:
            del history[:- (AVG_MOVE_LOOKBACK + 1)]

        ema = mmmid if prev_ema is None else EMA_ALPHA * mmmid + (1.0 - EMA_ALPHA) * float(prev_ema)
        fair = mmmid
        if len(history) >= 2:
            recent_moves = [abs(history[i] - history[i - 1]) for i in range(1, len(history))]
            avg_move = sum(recent_moves[-AVG_MOVE_LOOKBACK:]) / min(len(recent_moves), AVG_MOVE_LOOKBACK)
        else:
            avg_move = 2.0
        avg_move = max(MIN_AVG_MOVE, avg_move)

        deviation = mmmid - ema
        z_score = deviation / avg_move if avg_move > 0 else 0.0
        active_z = 0.0
        if abs(z_score) > MR_ZSCORE_THRESHOLD:
            clipped_z = max(-MR_ZSCORE_CLIP, min(MR_ZSCORE_CLIP, z_score))
            active_z = math.copysign(abs(clipped_z) - MR_ZSCORE_THRESHOLD, clipped_z)
            fair = mmmid - MR_SIGNAL_STRENGTH * active_z * avg_move

        memory["tomatoes_ema"] = ema
        memory["tomatoes_last_price"] = mmmid
        memory["tomatoes_last_fair"] = fair
        memory["tomatoes_avg_move"] = avg_move
        memory["tomatoes_z_score"] = z_score
        memory["tomatoes_active_z"] = active_z
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

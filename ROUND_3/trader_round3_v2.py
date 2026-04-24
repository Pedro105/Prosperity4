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


EMA_WINDOW = 5
EMA_ALPHA = 2.0 / (EMA_WINDOW + 1)
AVG_MOVE_LOOKBACK = 8
MR_SIGNAL_STRENGTH = 0.85
MR_ZSCORE_THRESHOLD = 0.50
MR_ZSCORE_CLIP = 3.0
MIN_AVG_MOVE = 1.0
TAKE_WIDTH = 0
CLEAR_WIDTH = 0
DEFAULT_EDGE = 0
JOIN_EDGE = 1
DISREGARD_EDGE = 0
PREVENT_ADVERSE = False
ADVERSE_VOLUME = 18

UNDERLYING = "VELVETFRUIT_EXTRACT"
OPTION_STRIKES = {
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
}
OPTION_LIMIT = 300
OPTION_SOFT_LIMIT = 30
OPTION_SIGMA = 0.23
OPTION_QUOTE_HALF_SPREAD = 2
OPTION_TAKE_EDGE = 3
OPTION_EXIT_EDGE = 2
OPTION_BASE_SIZE = 3
OPTION_INV_SKEW = 0.04
OPTION_HEDGE_BAND = 10_000
OPTION_HEDGE_EVERY = 500
OPTION_MAX_HEDGE_SIZE = 8
OPTION_PASSIVE_HEDGE_SKEW = 0.0
OPTION_MAX_PASSIVE_HEDGE_SHIFT = 0.0
OPTION_BUY_CUTOFF = 90_000
OPTION_FLATTEN_START = 95_000
OPTION_FLATTEN_EDGE = 4
OPTION_FLATTEN_SIZE = 6

CONFIG = {
    "VELVETFRUIT_EXTRACT": {
        "limit": 200,
        "spread_half": 1,
        "inventory_skew": 0.25,
        "base_size": 15,
        "min_size": 5,
        "size_position_step": 10,
        "take_width": TAKE_WIDTH,
        "clear_width": CLEAR_WIDTH,
        "default_edge": DEFAULT_EDGE,
        "join_edge": JOIN_EDGE,
        "disregard_edge": DISREGARD_EDGE,
        "prevent_adverse": PREVENT_ADVERSE,
        "adverse_volume": ADVERSE_VOLUME,
        "flatten_start": 95_000,
        "flatten_width": 0,
    },
    "HYDROGEL_PACK": {
        "limit": 200,
        "spread_half": 3,
        "inventory_skew": 0.35,
        "base_size": 8,
        "min_size": 1,
        "size_position_step": 5,
        "take_width": 1,
        "clear_width": 1,
        "default_edge": 1,
        "join_edge": 0,
        "disregard_edge": 1,
        "prevent_adverse": True,
        "adverse_volume": 6,
        "flatten_start": 80_000,
        "flatten_width": 1,
    },
}
 

class Trader:
    def run(self, state: TradingState):
        memory = self.load_memory(state.traderData)
        orders: Dict[str, List[Order]] = {product: [] for product in state.order_depths}
        option_delta = self.current_option_delta(state)
        memory["option_delta"] = round(option_delta, 4)

        for product, cfg in CONFIG.items():
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            position = int(state.position.get(product, 0))
            fair = self.random_walk_fair_value(product, depth, memory)
            if fair is None:
                continue
            if product == UNDERLYING:
                fair = self.apply_option_delta_skew(fair, option_delta)
            take_orders, buy_v, sell_v = self.take_orders(
                product,
                depth,
                fair,
                cfg["take_width"],
                position,
                cfg["limit"],
                cfg["prevent_adverse"],
                cfg["adverse_volume"],
            )
            clear_orders, buy_v, sell_v = self.clear_orders(
                product,
                depth,
                fair,
                cfg["clear_width"],
                position,
                buy_v,
                sell_v,
                cfg["limit"],
            )
            make_orders = self.make_orders(
                product,
                depth,
                fair,
                position,
                buy_v,
                sell_v,
                cfg["spread_half"],
                cfg["inventory_skew"],
                cfg["base_size"],
                cfg["min_size"],
                cfg["size_position_step"],
                cfg["disregard_edge"],
                cfg["join_edge"],
                cfg["default_edge"],
                cfg["limit"],
            )
            flatten_orders, buy_v, sell_v = self.late_flatten_orders(
                product,
                depth,
                fair,
                state.timestamp,
                position,
                buy_v,
                sell_v,
                cfg["limit"],
                cfg["flatten_start"],
                cfg["flatten_width"],
            )
            orders[product] = take_orders + clear_orders + make_orders + flatten_orders

            best_bid, best_ask = self.best_bid_ask(depth)
            print(
                f"[T={state.timestamp}] {product} pos={position} fair={fair:.2f} "
                f"best_bid={best_bid} best_ask={best_ask} ema={memory.get(self.mem_key(product, 'ema'), fair):.2f} "
                f"opt_delta={option_delta:.2f}"
            )

        self.trade_vouchers(state, orders)

        return orders, 0, json.dumps(memory, separators=(",", ":"))

    def load_memory(self, trader_data: str) -> Dict[str, object]:
        if not trader_data:
            return {}
        try:
            loaded = json.loads(trader_data)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    def best_bid(self, depth: Optional[OrderDepth]) -> Optional[int]:
        if depth is None or not depth.buy_orders:
            return None
        return max(depth.buy_orders)

    def best_ask(self, depth: Optional[OrderDepth]) -> Optional[int]:
        if depth is None or not depth.sell_orders:
            return None
        return min(depth.sell_orders)

    def best_bid_ask(self, depth: Optional[OrderDepth]) -> Tuple[Optional[int], Optional[int]]:
        return self.best_bid(depth), self.best_ask(depth)

    def mid_price(self, depth: Optional[OrderDepth]) -> Optional[float]:
        best_bid, best_ask = self.best_bid_ask(depth)
        if best_bid is None or best_ask is None or best_bid >= best_ask:
            return None
        return (best_bid + best_ask) / 2.0

    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call_price_delta(self, spot: float, strike: float, t: float, sigma: float) -> Tuple[float, float]:
        if t <= 0 or sigma <= 0:
            return max(0.0, spot - strike), 1.0 if spot > strike else 0.0
        vol_sqrt_t = sigma * math.sqrt(t)
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * t) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        price = spot * self.norm_cdf(d1) - strike * self.norm_cdf(d2)
        return max(0.0, price), self.norm_cdf(d1)

    def option_tte_years(self, timestamp: int) -> float:
        day_index = timestamp // 1_000_000
        tte_days = max(5.0, 8.0 - float(day_index))
        return tte_days / 365.0

    def current_option_delta(self, state: TradingState) -> float:
        underlying_depth = state.order_depths.get(UNDERLYING)
        spot = self.mid_price(underlying_depth)
        if spot is None:
            return 0.0

        total_delta = 0.0
        t = self.option_tte_years(state.timestamp)
        for product, strike in OPTION_STRIKES.items():
            position = int(state.position.get(product, 0))
            if position == 0:
                continue
            _, delta = self.bs_call_price_delta(spot, float(strike), t, OPTION_SIGMA)
            total_delta += position * delta
        return total_delta

    def apply_option_delta_skew(self, fair: float, option_delta: float) -> float:
        hedge_shift = max(
            -OPTION_MAX_PASSIVE_HEDGE_SHIFT,
            min(OPTION_MAX_PASSIVE_HEDGE_SHIFT, option_delta * OPTION_PASSIVE_HEDGE_SKEW),
        )
        return fair - hedge_shift

    def trade_vouchers(self, state: TradingState, orders: Dict[str, List[Order]]) -> None:
        underlying_depth = state.order_depths.get(UNDERLYING)
        spot = self.mid_price(underlying_depth)
        if spot is None:
            return

        t = self.option_tte_years(state.timestamp)
        option_delta = 0.0
        for product, strike in OPTION_STRIKES.items():
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            position = int(state.position.get(product, 0))
            fair, delta = self.bs_call_price_delta(spot, float(strike), t, OPTION_SIGMA)
            option_delta += position * delta
            option_orders = self.option_orders(product, depth, fair, position, state.timestamp)
            if option_orders:
                orders[product] = orders.get(product, []) + option_orders

            best_bid, best_ask = self.best_bid_ask(depth)
            print(
                f"[T={state.timestamp}] {product} pos={position} fair={fair:.2f} "
                f"best_bid={best_bid} best_ask={best_ask} delta={delta:.2f}"
            )

        self.hedge_option_delta(state, orders, option_delta)

    def option_orders(self, product: str, depth: OrderDepth, fair: float, position: int, timestamp: int) -> List[Order]:
        orders: List[Order] = []
        buy_cap = max(0, OPTION_SOFT_LIMIT - position)
        sell_cap = max(0, position)

        best_bid, best_ask = self.best_bid_ask(depth)
        if timestamp >= OPTION_FLATTEN_START and best_bid is not None and sell_cap > 0:
            if best_bid >= fair - OPTION_FLATTEN_EDGE:
                qty = min(OPTION_FLATTEN_SIZE, sell_cap, depth.buy_orders[best_bid])
                if qty > 0:
                    return [Order(product, best_bid, -qty)]

        can_add_gamma = timestamp < OPTION_BUY_CUTOFF
        if can_add_gamma and best_ask is not None and buy_cap > 0 and best_ask <= fair - OPTION_TAKE_EDGE:
            qty = min(OPTION_BASE_SIZE, buy_cap, -depth.sell_orders[best_ask])
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                buy_cap -= qty
        if best_bid is not None and sell_cap > 0 and best_bid >= fair + OPTION_EXIT_EDGE:
            qty = min(OPTION_BASE_SIZE, sell_cap, depth.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                sell_cap -= qty

        skew = OPTION_INV_SKEW * position
        raw_bid = int(math.floor(fair - OPTION_QUOTE_HALF_SPREAD - skew))
        raw_ask = int(math.ceil(fair + OPTION_QUOTE_HALF_SPREAD - skew))
        if can_add_gamma and buy_cap > 0 and (best_ask is None or raw_bid < best_ask):
            orders.append(Order(product, max(1, raw_bid), min(OPTION_BASE_SIZE, buy_cap)))
        if sell_cap > 0 and raw_ask >= fair + OPTION_EXIT_EDGE and (best_bid is None or raw_ask > best_bid):
            orders.append(Order(product, max(1, raw_ask), -min(OPTION_BASE_SIZE, sell_cap)))
        return orders

    def hedge_option_delta(self, state: TradingState, orders: Dict[str, List[Order]], option_delta: float) -> None:
        if state.timestamp % OPTION_HEDGE_EVERY != 0 or abs(option_delta) < OPTION_HEDGE_BAND:
            return
        depth = state.order_depths.get(UNDERLYING)
        best_bid, best_ask = self.best_bid_ask(depth)
        if depth is None or best_bid is None or best_ask is None:
            return

        vev_pos = int(state.position.get(UNDERLYING, 0))
        vev_limit = CONFIG[UNDERLYING]["limit"]
        hedge_orders = orders.get(UNDERLYING, [])

        hedge_needed = -int(round(option_delta))
        if hedge_needed < 0:
            qty = min(-hedge_needed, OPTION_MAX_HEDGE_SIZE, vev_limit + vev_pos)
            if qty > 0:
                hedge_orders.append(Order(UNDERLYING, best_bid, -qty))
        elif hedge_needed > 0:
            qty = min(hedge_needed, OPTION_MAX_HEDGE_SIZE, vev_limit - vev_pos)
            if qty > 0:
                hedge_orders.append(Order(UNDERLYING, best_ask, qty))

        orders[UNDERLYING] = hedge_orders

    def mem_key(self, product: str, suffix: str) -> str:
        return f"{product.lower()}_{suffix}"

    def clamp(self, value: int, low: int, high: int) -> int:
        return max(low, min(high, value))

    def random_walk_fair_value(self, product: str, depth: OrderDepth, memory: Dict[str, object]) -> Optional[float]:
        prev_ema = memory.get(self.mem_key(product, "ema"))
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
            if mm_ask is None or mm_bid is None or mm_bid >= mm_ask:
                mmmid = (best_bid + best_ask) / 2.0
            else:
                mmmid = (mm_bid + mm_ask) / 2.0

        history_key = self.mem_key(product, "mid_history")
        history = memory.setdefault(history_key, [])
        if isinstance(history, list):
            history.append(mmmid)
            if len(history) > AVG_MOVE_LOOKBACK + 1:
                del history[: -(AVG_MOVE_LOOKBACK + 1)]
        else:
            history = [mmmid]
            memory[history_key] = history

        ema = mmmid if prev_ema is None else EMA_ALPHA * mmmid + (1.0 - EMA_ALPHA) * float(prev_ema)
        fair = mmmid
        if len(history) >= 2:
            recent_moves = [abs(history[i] - history[i - 1]) for i in range(1, len(history))]
            avg_move = sum(recent_moves[-AVG_MOVE_LOOKBACK:]) / min(len(recent_moves), AVG_MOVE_LOOKBACK)
        else:
            avg_move = MIN_AVG_MOVE
        avg_move = max(MIN_AVG_MOVE, avg_move)

        deviation = mmmid - ema
        z_score = deviation / avg_move if avg_move > 0 else 0.0
        active_z = 0.0
        if abs(z_score) > MR_ZSCORE_THRESHOLD:
            clipped_z = max(-MR_ZSCORE_CLIP, min(MR_ZSCORE_CLIP, z_score))
            active_z = math.copysign(abs(clipped_z) - MR_ZSCORE_THRESHOLD, clipped_z)
            fair = mmmid - MR_SIGNAL_STRENGTH * active_z * avg_move

        memory[self.mem_key(product, "ema")] = ema
        memory[self.mem_key(product, "last_mid")] = mmmid
        memory[self.mem_key(product, "last_fair")] = fair
        memory[self.mem_key(product, "avg_move")] = avg_move
        memory[self.mem_key(product, "z_score")] = z_score
        memory[self.mem_key(product, "active_z")] = active_z
        return fair

    def take_orders(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        take_width: float,
        position: int,
        limit: int,
        prevent_adverse: bool,
        adverse_volume: int,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_v = 0
        sell_v = 0
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

    def clear_orders(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        clear_width: int,
        position: int,
        buy_v: int,
        sell_v: int,
        limit: int,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        pos_after_take = position + buy_v - sell_v
        bid_px = math.floor(fair - clear_width)
        ask_px = math.ceil(fair + clear_width)
        buy_cap = limit - (position + buy_v)
        sell_cap = limit + (position - sell_v)
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

    def late_flatten_orders(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        timestamp: int,
        position: int,
        buy_v: int,
        sell_v: int,
        limit: int,
        flatten_start: int,
        flatten_width: int,
    ) -> Tuple[List[Order], int, int]:
        if timestamp < flatten_start:
            return [], buy_v, sell_v

        orders: List[Order] = []
        pos_after_flow = position + buy_v - sell_v
        best_bid, best_ask = self.best_bid_ask(depth)
        bid_px = math.floor(fair - flatten_width)
        ask_px = math.ceil(fair + flatten_width)
        buy_cap = limit - (position + buy_v)
        sell_cap = limit + (position - sell_v)

        if pos_after_flow > 0 and best_bid is not None:
            sent = min(pos_after_flow, sell_cap, max(0, int(depth.buy_orders.get(best_bid, 0))))
            if sent > 0:
                price = max(best_bid, ask_px)
                orders.append(Order(product, price, -sent))
                sell_v += sent
        elif pos_after_flow < 0 and best_ask is not None:
            sent = min(abs(pos_after_flow), buy_cap, max(0, -int(depth.sell_orders.get(best_ask, 0))))
            if sent > 0:
                price = min(best_ask, bid_px)
                orders.append(Order(product, price, sent))
                buy_v += sent

        return orders, buy_v, sell_v

    def make_orders(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        position: int,
        buy_v: int,
        sell_v: int,
        spread_half: float,
        inventory_skew: float,
        base_size: int,
        min_size: int,
        size_position_step: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        limit: int,
    ) -> List[Order]:
        asks_above = [p for p in depth.sell_orders if p > fair + disregard_edge]
        bids_below = [p for p in depth.buy_orders if p < fair - disregard_edge]
        best_ask = min(asks_above) if asks_above else None
        best_bid = max(bids_below) if bids_below else None
        reservation = fair - inventory_skew * (position + buy_v - sell_v)
        raw_ask = math.ceil(reservation + spread_half + default_edge)
        raw_bid = math.floor(reservation - spread_half - default_edge)
        ask = raw_ask if best_ask is None else (best_ask if abs(best_ask - reservation) <= join_edge else best_ask - 1)
        bid = raw_bid if best_bid is None else (best_bid if abs(reservation - best_bid) <= join_edge else best_bid + 1)
        if best_ask is not None:
            ask = min(max(ask, raw_ask), best_ask)
        if best_bid is not None:
            bid = max(min(bid, raw_bid), best_bid)
        if bid >= ask:
            bid = ask - 1
        orders: List[Order] = []
        buy_qty = limit - (position + buy_v)
        sell_qty = limit + (position - sell_v)
        bid_size = self.clamp(base_size - max(position + buy_v - sell_v, 0) // size_position_step, min_size, buy_qty)
        ask_size = self.clamp(base_size + min(position + buy_v - sell_v, 0) // size_position_step, min_size, sell_qty)
        if buy_qty > 0 and bid_size > 0:
            orders.append(Order(product, int(math.floor(bid)), bid_size))
        if sell_qty > 0 and ask_size > 0:
            orders.append(Order(product, int(math.ceil(ask)), -ask_size))
        return orders

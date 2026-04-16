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

# -----------------------------------------------------------------------------
# PEPPER_ROOT parameters (UNCHANGED from trader_round1.py)
# -----------------------------------------------------------------------------
# Derived from simple linear regression on day -2, -1, 0 mid-prices.
PEPPER_SLOPE = 0.0010000612960100698
PEPPER_HALF_SPREAD = 2
PEPPER_BUY_EDGE = 10
PEPPER_SELL_EDGE = 1
PEPPER_SIZE = 20
PEPPER_LATE_START = 995_000
PEPPER_FINAL_UNWIND = 1000000

# -----------------------------------------------------------------------------
# OSMIUM: market-making-bot-aware fair value model (tunable)
# -----------------------------------------------------------------------------
# Maker layer detection
OSMIUM_MAKER_SIZE_MIN = 20
OSMIUM_MAKER_SIZE_MAX = 30
# Absolute size difference allowed to call the layer "symmetric"
OSMIUM_MAKER_SYMMETRY_TOL = 2
# How far from touch we search/allow (1=touch only, 2=prefer L2, 3=up to 3 levels)
OSMIUM_MAX_LEVEL = 3
# Prefer deeper large layer (L2) consistent with observed behavior
OSMIUM_L2_BONUS = 1.5
# Penalize layers further from touch (small penalty so L2 can still win)
OSMIUM_LEVEL_DISTANCE_PENALTY = 0.4

# Fair value smoothing/robustness
OSMIUM_MAKER_MID_EMA_ALPHA = 0.15  # smooth maker-mid to reduce flicker
OSMIUM_MAKER_CONFIRM_TTL = 3  # keep last detected maker layer for a few ticks if book is sparse

# Fallback fair model (matches existing trader_round1.py)
OSMIUM_BETA = -0.4952
OSMIUM_EMA_ALPHA = 0.05
OSMIUM_FALLBACK_MODE = "predicted"  # "predicted" (default) or "mid"

# Trading parameters around fair
OSMIUM_QUOTE_HALF_SPREAD = 1
OSMIUM_TAKE_THRESHOLD = 0  # take if price is clearly better than fair +/- threshold
OSMIUM_INV_SKEW = 0.03

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

    def quote_prices(
        self, fair: float, half_spread: float, inv_skew: float, position: int, depth: OrderDepth
    ) -> Tuple[int, int]:
        reservation = fair - inv_skew * position
        raw_bid = math.floor(reservation - half_spread)
        raw_ask = math.ceil(reservation + half_spread)
        best_bid, best_ask = self.best_bid_ask(depth)
        bid = raw_bid if best_bid is None else min(raw_bid, best_bid + 1)
        ask = raw_ask if best_ask is None else max(raw_ask, best_ask - 1)
        if bid >= ask:
            bid = ask - 1
        return bid, ask

    def take_orders(
        self, product: str, fair: float, edge: float, depth: OrderDepth, position: int
    ) -> Tuple[List[Order], int]:
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

    def make_orders(
        self,
        product: str,
        fair: float,
        depth: OrderDepth,
        position: int,
        net_after_take: int,
        half_spread: float,
        inv_skew: float,
    ) -> List[Order]:
        bid, ask = self.quote_prices(fair, half_spread, inv_skew, position + net_after_take, depth)
        buy_cap = LIMITS[product] - (position + max(net_after_take, 0))
        sell_cap = LIMITS[product] + (position + min(net_after_take, 0))
        orders: List[Order] = []
        if buy_cap > 0:
            orders.append(Order(product, bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, ask, -sell_cap))
        return orders

    # -------------------------------------------------------------------------
    # PEPPER_ROOT strategy (PRESERVED EXACTLY)
    # -------------------------------------------------------------------------
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
            target_position = int(limit)
        elif timestamp < PEPPER_FINAL_UNWIND:
            target_position = 0
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
            ask_qty = min( new_position - target_position, sell_cap)
            if ask_qty > 0:
                orders.append(Order(PEPPER, ask_quote, -ask_qty))

        return orders

    # -------------------------------------------------------------------------
    # OSMIUM: maker-aware fair value and trading
    # -------------------------------------------------------------------------
    def _osmium_ladders(self, depth: OrderDepth, max_levels: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        bids = []
        if depth.buy_orders:
            for p in sorted(depth.buy_orders.keys(), reverse=True)[:max_levels]:
                bids.append((int(p), int(depth.buy_orders[p])))
        asks = []
        if depth.sell_orders:
            for p in sorted(depth.sell_orders.keys())[:max_levels]:
                asks.append((int(p), int(-depth.sell_orders[p])))
        return bids, asks

    def detect_osmium_maker_layer(self, depth: OrderDepth, memory: Dict[str, float]) -> Optional[Dict[str, float]]:
        bids, asks = self._osmium_ladders(depth, OSMIUM_MAX_LEVEL)
        if not bids or not asks:
            return None

        best_bid = bids[0][0]
        best_ask = asks[0][0]

        def in_maker_range(qty: int) -> bool:
            return OSMIUM_MAKER_SIZE_MIN <= qty <= OSMIUM_MAKER_SIZE_MAX

        best: Optional[Dict[str, float]] = None
        best_score = float("-inf")

        max_pair_levels = min(len(bids), len(asks), OSMIUM_MAX_LEVEL)
        for i in range(max_pair_levels):
            bid_p, bid_q = bids[i]
            ask_p, ask_q = asks[i]
            level = i + 1

            score = 0.0
            if in_maker_range(bid_q):
                score += 2.0
            if in_maker_range(ask_q):
                score += 2.0

            # Symmetry is a strong hint of one dominant maker on both sides.
            if in_maker_range(bid_q) and in_maker_range(ask_q):
                if abs(bid_q - ask_q) <= OSMIUM_MAKER_SYMMETRY_TOL:
                    score += 2.5
                else:
                    score += 0.5

            # Prefer not-too-far from touch; still allow L2 to win.
            score -= OSMIUM_LEVEL_DISTANCE_PENALTY * (level - 1)

            # Prefer the observed "one level behind touch" behavior.
            if level == 2:
                score += OSMIUM_L2_BONUS

            # Sanity: maker layer should not be wildly off-touch.
            if bid_p < best_bid - 10 * level or ask_p > best_ask + 10 * level:
                score -= 2.0

            if score > best_score:
                best_score = score
                best = {
                    "maker_level": float(level),
                    "maker_bid": float(bid_p),
                    "maker_ask": float(ask_p),
                    "maker_bid_qty": float(bid_q),
                    "maker_ask_qty": float(ask_q),
                    "maker_score": float(score),
                }

        # Require two-sided evidence (avoid hallucinating fair from one side).
        if best is None:
            return None
        if best["maker_bid"] >= best["maker_ask"]:
            return None

        # If the best layer doesn't have at least one side in maker range, it's not a maker signal.
        if not (in_maker_range(int(best["maker_bid_qty"])) or in_maker_range(int(best["maker_ask_qty"]))):
            return None

        return best

    def _fallback_osmium_fair(self, depth: OrderDepth, memory: Dict[str, float]) -> Optional[float]:
        mid = self.mid_price(depth)
        if mid is None:
            return memory.get("osmium_fair")

        if OSMIUM_FALLBACK_MODE == "mid":
            memory["osmium_fair"] = float(mid)
            return float(mid)

        prev_mean = memory.get("osmium_mean")
        mean = mid if prev_mean is None else OSMIUM_EMA_ALPHA * mid + (1.0 - OSMIUM_EMA_ALPHA) * prev_mean
        predicted = mid + OSMIUM_BETA * (mid - mean)
        memory["osmium_mean"] = float(mean)
        memory["osmium_fair"] = float(predicted)
        return float(predicted)

    def compute_osmium_fair_value(self, depth: OrderDepth, memory: Dict[str, float]) -> Optional[float]:
        detected = self.detect_osmium_maker_layer(depth, memory)
        if detected is None:
            # If we recently had a confirmed maker layer, keep it briefly to avoid flicker on sparse books.
            ttl = int(memory.get("osmium_maker_ttl", 0))
            if ttl > 0:
                memory["osmium_maker_ttl"] = ttl - 1
                last_mid = memory.get("osmium_maker_mid")
                if last_mid is not None:
                    memory["osmium_fair"] = float(last_mid)
                    return float(last_mid)
            return self._fallback_osmium_fair(depth, memory)

        maker_bid = float(detected["maker_bid"])
        maker_ask = float(detected["maker_ask"])
        maker_mid = (maker_bid + maker_ask) / 2.0

        # Smooth maker mid slightly so small book reshuffles don't whipsaw fair.
        prev = memory.get("osmium_maker_mid")
        smoothed = maker_mid if prev is None else (OSMIUM_MAKER_MID_EMA_ALPHA * maker_mid + (1.0 - OSMIUM_MAKER_MID_EMA_ALPHA) * float(prev))

        memory["osmium_maker_mid"] = float(smoothed)
        memory["osmium_maker_bid"] = float(maker_bid)
        memory["osmium_maker_ask"] = float(maker_ask)
        memory["osmium_maker_level"] = float(detected["maker_level"])
        memory["osmium_maker_score"] = float(detected["maker_score"])
        memory["osmium_maker_ttl"] = int(OSMIUM_MAKER_CONFIRM_TTL)

        # Use maker-mid as fair (less noisy than top-of-book mid).
        memory["osmium_fair"] = float(smoothed)
        return float(smoothed)

    def trade_osmium(self, depth: OrderDepth, position: int, memory: Dict[str, float]) -> List[Order]:
        fair = self.compute_osmium_fair_value(depth, memory)
        if fair is None:
            return []

        take, net = self.take_orders(OSMIUM, fair, OSMIUM_TAKE_THRESHOLD, depth, position)
        make = self.make_orders(OSMIUM, fair, depth, position, net, OSMIUM_QUOTE_HALF_SPREAD, OSMIUM_INV_SKEW)
        return take + make

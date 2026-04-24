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
PEPPER_SLOPE = 0.0012722395322395
PEPPER_HALF_SPREAD = 2
PEPPER_BUY_EDGE = 10
PEPPER_SELL_EDGE = 1
PEPPER_SIZE = 20
PEPPER_LATE_START = 75_000
PEPPER_FINAL_UNWIND = 98_500

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
OSMIUM_MAKER_CONFIRM_TTL = 1  # keep last detected maker layer for a few ticks if book is sparse
OSMIUM_MAKER_SCORE_MIN = 3.0
OSMIUM_STRONG_MAKER_SCORE = 5.5
OSMIUM_VERY_STRONG_MAKER_SCORE = 7.0

# Fallback fair model (matches existing trader_round1.py)
OSMIUM_BETA = -0.4952
OSMIUM_EMA_ALPHA = 0.05
OSMIUM_FALLBACK_MODE = "predicted"  # "predicted" (default) or "mid"

# Trading parameters around fair
OSMIUM_QUOTE_HALF_SPREAD = 1
OSMIUM_TAKE_THRESHOLD = 0  # take if price is clearly better than fair +/- threshold
OSMIUM_INV_SKEW = 0.03
OSMIUM_STRONG_QUOTE_HALF_SPREAD = 1.0
OSMIUM_VERY_STRONG_QUOTE_HALF_SPREAD = 0.5
OSMIUM_WEAK_QUOTE_HALF_SPREAD = 1.5
OSMIUM_FALLBACK_QUOTE_HALF_SPREAD = 3.0
OSMIUM_STRONG_TAKE_THRESHOLD = 0
OSMIUM_WEAK_TAKE_THRESHOLD = 1
OSMIUM_FALLBACK_TAKE_THRESHOLD = 2
OSMIUM_STRONG_INV_SKEW = 0.02
OSMIUM_WEAK_INV_SKEW = 0.03
OSMIUM_FALLBACK_INV_SKEW = 0.05
OSMIUM_INVENTORY_FAIR_SHIFT = 0.05
OSMIUM_VERY_STRONG_QUOTE_SIZE = 50
OSMIUM_STRONG_QUOTE_SIZE = 40
OSMIUM_WEAK_QUOTE_SIZE = 25
OSMIUM_FALLBACK_QUOTE_SIZE = 15

LIMITS = {PEPPER: 80, OSMIUM: 80}


class Trader:
    def bid(self) -> int:
        # MAF bid for Round 2 extra market access.
        # Keep the bid inside the EV-positive range but slightly above
        # the most conservative floor to improve acceptance odds.
        return 1000

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
            for ask in sorted(depth.sell_orders):
                ask_qty = -depth.sell_orders[ask]
                if ask > fair - edge:
                    break
                current_position = position + net
                if current_position >= limit:
                    break
                qty = min(ask_qty, limit - current_position)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    net += qty
        if depth.buy_orders:
            for bid in sorted(depth.buy_orders, reverse=True):
                bid_qty = depth.buy_orders[bid]
                if bid < fair + edge:
                    break
                current_position = position + net
                if current_position <= -limit:
                    break
                qty = min(bid_qty, limit + current_position)
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

    def make_orders_with_size_cap(
        self,
        product: str,
        fair: float,
        depth: OrderDepth,
        position: int,
        net_after_take: int,
        half_spread: float,
        inv_skew: float,
        quote_size: int,
    ) -> List[Order]:
        bid, ask = self.quote_prices(fair, half_spread, inv_skew, position + net_after_take, depth)
        buy_cap = LIMITS[product] - (position + max(net_after_take, 0))
        sell_cap = LIMITS[product] + (position + min(net_after_take, 0))
        orders: List[Order] = []
        if buy_cap > 0:
            orders.append(Order(product, bid, min(buy_cap, quote_size)))
        if sell_cap > 0:
            orders.append(Order(product, ask, -min(sell_cap, quote_size)))
        return orders

    # -------------------------------------------------------------------------
    # PEPPER_ROOT strategy (PRESERVED EXACTLY)
    # -------------------------------------------------------------------------
    def pepper_target_position(self, timestamp: int, limit: int) -> int:
        if timestamp < PEPPER_LATE_START:
            return limit
        if timestamp >= PEPPER_FINAL_UNWIND:
            return 0
        unwind_window = max(1, PEPPER_FINAL_UNWIND - PEPPER_LATE_START)
        remaining = PEPPER_FINAL_UNWIND - timestamp
        return max(0, min(limit, math.ceil(limit * remaining / unwind_window)))

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
        target_position = self.pepper_target_position(timestamp, limit)

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
        touch_width = best_ask - best_bid
        for bid_level, (bid_p, bid_q) in enumerate(bids, start=1):
            for ask_level, (ask_p, ask_q) in enumerate(asks, start=1):
                if bid_p >= ask_p:
                    continue
                bid_in_range = in_maker_range(bid_q)
                ask_in_range = in_maker_range(ask_q)
                if not (bid_in_range or ask_in_range):
                    continue

                score = 0.0
                if bid_in_range:
                    score += 2.0
                if ask_in_range:
                    score += 2.0

                # Symmetry is a strong hint of one dominant maker on both sides.
                if bid_in_range and ask_in_range:
                    if abs(bid_q - ask_q) <= OSMIUM_MAKER_SYMMETRY_TOL:
                        score += 2.5
                    else:
                        score += 1.0

                score -= OSMIUM_LEVEL_DISTANCE_PENALTY * ((bid_level - 1) + (ask_level - 1)) / 2.0
                if bid_level == 2:
                    score += OSMIUM_L2_BONUS * 0.5
                if ask_level == 2:
                    score += OSMIUM_L2_BONUS * 0.5
                score -= 0.25 * abs(bid_level - ask_level)

                width = ask_p - bid_p
                if width > touch_width:
                    score -= 0.10 * (width - touch_width)

                # Sanity: maker layer should not be wildly off-touch.
                if bid_p < best_bid - 10 * bid_level or ask_p > best_ask + 10 * ask_level:
                    score -= 2.0

                if score > best_score:
                    best_score = score
                    best = {
                        "maker_level": float((bid_level + ask_level) / 2.0),
                        "maker_bid_level": float(bid_level),
                        "maker_ask_level": float(ask_level),
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
        if best["maker_score"] < OSMIUM_MAKER_SCORE_MIN:
            return None

        return best

    def _fallback_osmium_fair(self, depth: OrderDepth, memory: Dict[str, float]) -> Optional[float]:
        mid = self.mid_price(depth)
        if mid is None:
            return memory.get("osmium_fair")

        if OSMIUM_FALLBACK_MODE == "mid":
            memory["osmium_fair"] = float(mid)
            return float(mid)

        bids, asks = self._osmium_ladders(depth, OSMIUM_MAX_LEVEL)
        sparse_book = len(bids) < 2 or len(asks) < 2
        prev_mean = memory.get("osmium_mean")
        mean = mid if prev_mean is None else OSMIUM_EMA_ALPHA * mid + (1.0 - OSMIUM_EMA_ALPHA) * prev_mean
        predicted = mid + OSMIUM_BETA * (mid - mean)
        last_maker = memory.get("osmium_maker_mid")
        fair = max(predicted, float(last_maker)) if sparse_book and last_maker is not None else predicted
        memory["osmium_mean"] = float(mean)
        memory["osmium_fair"] = float(fair)
        return float(fair)

    def compute_osmium_fair_value(self, depth: OrderDepth, memory: Dict[str, float]) -> Optional[float]:
        detected = self.detect_osmium_maker_layer(depth, memory)
        if detected is None:
            memory["osmium_live_maker"] = 0.0
            memory["osmium_regime"] = "fallback"
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
        predicted = self._fallback_osmium_fair(depth, memory)

        # Smooth maker mid slightly so small book reshuffles don't whipsaw fair.
        prev = memory.get("osmium_maker_mid")
        smoothed = maker_mid if prev is None else (OSMIUM_MAKER_MID_EMA_ALPHA * maker_mid + (1.0 - OSMIUM_MAKER_MID_EMA_ALPHA) * float(prev))
        maker_score = float(detected["maker_score"])
        fair = smoothed if maker_score >= OSMIUM_MAKER_SCORE_MIN else predicted

        memory["osmium_maker_mid"] = float(smoothed)
        memory["osmium_maker_bid"] = float(maker_bid)
        memory["osmium_maker_ask"] = float(maker_ask)
        memory["osmium_maker_level"] = float(detected["maker_level"])
        memory["osmium_maker_bid_level"] = float(detected["maker_bid_level"])
        memory["osmium_maker_ask_level"] = float(detected["maker_ask_level"])
        memory["osmium_maker_score"] = maker_score
        memory["osmium_maker_ttl"] = int(OSMIUM_MAKER_CONFIRM_TTL)
        memory["osmium_live_maker"] = 1.0
        memory["osmium_regime"] = "strong" if maker_score >= OSMIUM_STRONG_MAKER_SCORE else "weak"

        # Blend maker-mid with the fallback predictor to reduce overreaction.
        memory["osmium_fair"] = float(fair)
        return float(fair)

    def osmium_regime_params(self, memory: Dict[str, float]) -> Tuple[float, float, float, int]:
        regime = memory.get("osmium_regime", "fallback")
        maker_score = float(memory.get("osmium_maker_score", 0.0))
        if regime == "strong" and maker_score >= OSMIUM_VERY_STRONG_MAKER_SCORE:
            return (
                OSMIUM_VERY_STRONG_QUOTE_HALF_SPREAD,
                OSMIUM_STRONG_TAKE_THRESHOLD,
                OSMIUM_STRONG_INV_SKEW,
                OSMIUM_VERY_STRONG_QUOTE_SIZE,
            )
        if regime == "strong":
            return (
                OSMIUM_STRONG_QUOTE_HALF_SPREAD,
                OSMIUM_STRONG_TAKE_THRESHOLD,
                OSMIUM_STRONG_INV_SKEW,
                OSMIUM_STRONG_QUOTE_SIZE,
            )
        if regime == "weak":
            return (
                OSMIUM_WEAK_QUOTE_HALF_SPREAD,
                OSMIUM_WEAK_TAKE_THRESHOLD,
                OSMIUM_WEAK_INV_SKEW,
                OSMIUM_WEAK_QUOTE_SIZE,
            )
        return (
            OSMIUM_FALLBACK_QUOTE_HALF_SPREAD,
            OSMIUM_FALLBACK_TAKE_THRESHOLD,
            OSMIUM_FALLBACK_INV_SKEW,
            OSMIUM_FALLBACK_QUOTE_SIZE,
        )

    def osmium_inventory_adjusted_fair(self, fair: float, position: int, memory: Dict[str, float]) -> float:
        regime = memory.get("osmium_regime", "fallback")
        shift = OSMIUM_INVENTORY_FAIR_SHIFT
        if regime == "strong":
            shift *= 1.5
        elif regime == "fallback":
            shift *= 0.5
        return fair - shift * position

    def trade_osmium(self, depth: OrderDepth, position: int, memory: Dict[str, float]) -> List[Order]:
        fair = self.compute_osmium_fair_value(depth, memory)
        if fair is None:
            return []

        quote_half_spread, take_threshold, inv_skew, quote_size = self.osmium_regime_params(memory)
        adjusted_fair = self.osmium_inventory_adjusted_fair(fair, position, memory)
        take, net = self.take_orders(OSMIUM, adjusted_fair, take_threshold, depth, position)
        make = self.make_orders_with_size_cap(
            OSMIUM, adjusted_fair, depth, position, net, quote_half_spread, inv_skew, quote_size
        )
        return take + make


if __name__ == "__main__":
    print("`trader_MMbot.py` is the uploadable core trader and does not print local diagnostics.")
    print("Run `python3 ROUND_1/trader_MMbot_local.py` to execute the local backtest, print stats, and generate plots.")
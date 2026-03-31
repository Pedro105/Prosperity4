from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Any
import json
import math


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
    },
    "TOMATOES": {
        "history_len": 64,
        "ema_alpha": 0.20,
        "fast_ema_alpha": 0.35,
        "slow_ema_alpha": 0.08,
        "min_avg_move": 1.5,
        "avg_move_lookback": 16,
        "flatten_threshold": 50,
        "take_inventory_skew": 0.07,
        "reservation_inventory_skew": 0.15,
        "base_size": 20,
        "min_size": 6,
        "size_position_step": 5,
    },
}


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    DEFAULT_STATE = {
        "mid_history": {"TOMATOES": []},
        "ema": {"TOMATOES": None},
        "fast_ema": {"TOMATOES": None},
        "slow_ema": {"TOMATOES": None},
    }

    def run(self, state: TradingState):
        memory = self.load_state(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)

            if product == "EMERALDS":
                result[product] = self.trade_emeralds(order_depth, position)
            elif product == "TOMATOES":
                result[product] = self.trade_tomatoes(order_depth, position, memory)

        trader_data = json.dumps(memory, separators=(",", ":"))
        return result, 0, trader_data

    def load_state(self, trader_data: str) -> Dict[str, Any]:
        if not trader_data:
            return json.loads(json.dumps(self.DEFAULT_STATE))
        try:
            loaded = json.loads(trader_data)
            for key, default in self.DEFAULT_STATE.items():
                if key not in loaded:
                    loaded[key] = json.loads(json.dumps(default))
                if "TOMATOES" not in loaded[key]:
                    loaded[key]["TOMATOES"] = default["TOMATOES"]
            return loaded
        except Exception:
            return json.loads(json.dumps(self.DEFAULT_STATE))

    def cfg(self, product: str, key: str):
        return PARAMS[product][key]

    def clamp(self, value: int, low: int, high: int) -> int:
        return max(low, min(high, value))

    def best_bid_ask(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def microprice(self, order_depth: OrderDepth):
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return None

        bid_vol = order_depth.buy_orders.get(best_bid, 0)
        ask_vol = -order_depth.sell_orders.get(best_ask, 0)
        if bid_vol <= 0 or ask_vol <= 0:
            return (best_bid + best_ask) / 2
        return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

    def take_liquidity_buy(
        self, product: str, orders: List[Order], order_depth: OrderDepth, max_price: int, position: int, limit: int
    ) -> int:
        for ask in sorted(order_depth.sell_orders.keys()):
            if ask > max_price:
                break
            qty = min(-order_depth.sell_orders[ask], limit - position)
            if qty > 0:
                orders.append(Order(product, ask, qty))
                position += qty
        return position

    def take_liquidity_sell(
        self, product: str, orders: List[Order], order_depth: OrderDepth, min_price: int, position: int, limit: int
    ) -> int:
        for bid in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid < min_price:
                break
            qty = min(order_depth.buy_orders[bid], limit + position)
            if qty > 0:
                orders.append(Order(product, bid, -qty))
                position -= qty
        return position

    def trade_emeralds(self, order_depth: OrderDepth, position: int) -> List[Order]:
        product = "EMERALDS"
        limit = self.POSITION_LIMITS[product]
        fair_value = self.cfg(product, "fair_value")
        take_edge = self.cfg(product, "take_edge")
        orders: List[Order] = []

        best_bid, best_ask = self.best_bid_ask(order_depth)

        position = self.take_liquidity_buy(product, orders, order_depth, fair_value - take_edge, position, limit)
        position = self.take_liquidity_sell(product, orders, order_depth, fair_value + take_edge, position, limit)

        flatten_threshold = self.cfg(product, "flatten_threshold")
        if position > flatten_threshold:
            position = self.take_liquidity_sell(product, orders, order_depth, fair_value, position, limit)
        elif position < -flatten_threshold:
            position = self.take_liquidity_buy(product, orders, order_depth, fair_value, position, limit)

        reservation = fair_value - self.cfg(product, "inventory_skew") * position
        raw_bid = math.floor(reservation - self.cfg(product, "spread_half"))
        raw_ask = math.ceil(reservation + self.cfg(product, "spread_half"))

        bid_quote = min(raw_bid, best_bid + 1) if best_bid is not None else raw_bid
        ask_quote = max(raw_ask, best_ask - 1) if best_ask is not None else raw_ask
        if bid_quote >= ask_quote:
            bid_quote = ask_quote - 1

        buy_cap = limit - position
        sell_cap = limit + position
        base_size = self.cfg(product, "base_size")
        min_size = self.cfg(product, "min_size")
        step = self.cfg(product, "size_position_step")

        bid_size = self.clamp(base_size - max(position, 0) // step, min_size, buy_cap)
        ask_size = self.clamp(base_size + min(position, 0) // step, min_size, sell_cap)

        if buy_cap > 0:
            orders.append(Order(product, bid_quote, bid_size))
        if sell_cap > 0:
            orders.append(Order(product, ask_quote, -ask_size))
        return orders

    def build_tomatoes_features(
        self,
        order_depth: OrderDepth,
        history: List[float],
        mid: float,
        micro: float,
        fast_ema: float,
        slow_ema: float,
    ) -> Dict[str, float]:
        """
        Build a compact online feature table for TOMATOES.

        Why these features:
        - spread: execution quality / market tightness
        - micro_minus_mid: best-level pressure, useful for short-term drift
        - imbalance_l1 / imbalance_l3: whether bids or asks dominate the book
        - short_return / medium_return: captures local momentum vs snap-back
        - avg_move: realized local volatility, needed to scale thresholds
        - trend_gap: fast-vs-slow EMA difference, a simple trend proxy

        These are chosen because they are cheap to compute inside the live
        trading loop and map naturally to regime detection without training.
        """
        best_bid, best_ask = self.best_bid_ask(order_depth)
        spread = 0.0 if best_bid is None or best_ask is None else float(best_ask - best_bid)

        bid_items = sorted(order_depth.buy_orders.items(), reverse=True)
        ask_items = sorted(order_depth.sell_orders.items())

        bid_l1 = bid_items[0][1] if bid_items else 0
        ask_l1 = -ask_items[0][1] if ask_items else 0

        bid_l3 = sum(vol for _, vol in bid_items[:3])
        ask_l3 = sum(-vol for _, vol in ask_items[:3])

        def imbalance(bid_volume: float, ask_volume: float) -> float:
            total = bid_volume + ask_volume
            return 0.0 if total <= 0 else (bid_volume - ask_volume) / total

        short_return = 0.0 if len(history) < 5 else mid - history[-5]
        medium_return = 0.0 if len(history) < 12 else mid - history[-12]

        if len(history) >= 2:
            moves = [abs(history[i] - history[i - 1]) for i in range(1, len(history))]
            lookback = self.cfg("TOMATOES", "avg_move_lookback")
            avg_move = sum(moves[-lookback:]) / min(len(moves), lookback)
        else:
            avg_move = 0.0

        return {
            "spread": spread,
            "micro_minus_mid": micro - mid,
            "imbalance_l1": imbalance(float(bid_l1), float(ask_l1)),
            "imbalance_l3": imbalance(float(bid_l3), float(ask_l3)),
            "short_return": short_return,
            "medium_return": medium_return,
            "avg_move": max(self.cfg("TOMATOES", "min_avg_move"), avg_move),
            "trend_gap": fast_ema - slow_ema,
        }

    def detect_tomatoes_regime(self, features: Dict[str, float]) -> str:
        """
        A heuristic regime detector is a natural first step here because the
        dataset is small and noisy. This avoids overfitting a heavier ML model
        while still letting the strategy behave differently in different market
        conditions.
        """
        avg_move = features["avg_move"]
        trend_gap = features["trend_gap"]
        imbalance = features["imbalance_l1"]
        micro_shift = features["micro_minus_mid"]

        if abs(trend_gap) > 1.2 * avg_move and abs(features["medium_return"]) > 2.0 * avg_move:
            return "trend"

        if avg_move > 4.0 and abs(trend_gap) < 0.8 * avg_move:
            return "volatile"

        if abs(micro_shift) < 0.8 and abs(imbalance) < 0.18 and abs(features["short_return"]) < 1.5 * avg_move:
            return "mean_revert"

        return "balanced"

    def trade_tomatoes(self, order_depth: OrderDepth, position: int, memory: Dict[str, Any]) -> List[Order]:
        product = "TOMATOES"
        limit = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders

        mid = (best_bid + best_ask) / 2
        micro = self.microprice(order_depth)
        if micro is None:
            micro = mid

        history = memory["mid_history"]["TOMATOES"]
        history.append(mid)
        history_len = self.cfg(product, "history_len")
        if len(history) > history_len:
            history[:] = history[-history_len:]

        ema_alpha = self.cfg(product, "ema_alpha")
        prev_ema = memory["ema"]["TOMATOES"]
        ema = mid if prev_ema is None else ema_alpha * mid + (1 - ema_alpha) * prev_ema
        memory["ema"]["TOMATOES"] = ema

        fast_alpha = self.cfg(product, "fast_ema_alpha")
        prev_fast = memory["fast_ema"]["TOMATOES"]
        fast_ema = mid if prev_fast is None else fast_alpha * mid + (1 - fast_alpha) * prev_fast
        memory["fast_ema"]["TOMATOES"] = fast_ema

        slow_alpha = self.cfg(product, "slow_ema_alpha")
        prev_slow = memory["slow_ema"]["TOMATOES"]
        slow_ema = mid if prev_slow is None else slow_alpha * mid + (1 - slow_alpha) * prev_slow
        memory["slow_ema"]["TOMATOES"] = slow_ema

        features = self.build_tomatoes_features(order_depth, history, mid, micro, fast_ema, slow_ema)
        regime = self.detect_tomatoes_regime(features)

        # Feature table is converted into a fair value differently per regime.
        # Mean reversion keeps price near micro/EMA. Trend leans with fast-vs-slow
        # momentum. Volatile widens the edges to avoid being picked off.
        fair_value = 0.70 * micro + 0.30 * ema
        entry_edge = max(3.0, 1.25 * features["avg_move"])
        quote_edge = max(1.0, 0.55 * features["avg_move"])

        if regime == "trend":
            fair_value += 0.55 * features["trend_gap"] + 1.25 * features["micro_minus_mid"]
            entry_edge += 0.25 * features["avg_move"]
            quote_edge += 0.20 * features["avg_move"]
        elif regime == "volatile":
            fair_value += 0.20 * features["micro_minus_mid"]
            entry_edge += 0.70 * features["avg_move"]
            quote_edge += 0.45 * features["avg_move"]
        elif regime == "mean_revert":
            fair_value += 0.35 * features["micro_minus_mid"] - 0.15 * features["short_return"]
        else:
            fair_value += 0.25 * features["micro_minus_mid"]

        take_skew = self.cfg(product, "take_inventory_skew")
        take_buy_price = math.floor(fair_value - entry_edge - take_skew * position)
        take_sell_price = math.ceil(fair_value + entry_edge - take_skew * position)

        position = self.take_liquidity_buy(product, orders, order_depth, take_buy_price, position, limit)
        position = self.take_liquidity_sell(product, orders, order_depth, take_sell_price, position, limit)

        flatten_threshold = self.cfg(product, "flatten_threshold")
        if position > flatten_threshold:
            position = self.take_liquidity_sell(product, orders, order_depth, math.floor(fair_value), position, limit)
        elif position < -flatten_threshold:
            position = self.take_liquidity_buy(product, orders, order_depth, math.ceil(fair_value), position, limit)

        reservation = fair_value - self.cfg(product, "reservation_inventory_skew") * position
        raw_bid = math.floor(reservation - quote_edge)
        raw_ask = math.ceil(reservation + quote_edge)

        bid_quote = min(raw_bid, best_bid + 1)
        ask_quote = max(raw_ask, best_ask - 1)
        if bid_quote >= ask_quote:
            bid_quote = ask_quote - 1

        buy_cap = limit - position
        sell_cap = limit + position
        base_size = self.cfg(product, "base_size")
        min_size = self.cfg(product, "min_size")
        step = self.cfg(product, "size_position_step")

        # In a trend we lean with the move by allowing slightly more size on the
        # favorable side. In volatility we reduce sizes because passive fills are
        # more likely to be adverse.
        bid_size = self.clamp(base_size - max(position, 0) // step, min_size, buy_cap)
        ask_size = self.clamp(base_size + min(position, 0) // step, min_size, sell_cap)
        if regime == "trend":
            if features["trend_gap"] > 0:
                bid_size = self.clamp(bid_size + 2, min_size, buy_cap)
                ask_size = self.clamp(ask_size - 2, min_size, sell_cap)
            else:
                ask_size = self.clamp(ask_size + 2, min_size, sell_cap)
                bid_size = self.clamp(bid_size - 2, min_size, buy_cap)
        elif regime == "volatile":
            bid_size = self.clamp(bid_size - 2, min_size, buy_cap)
            ask_size = self.clamp(ask_size - 2, min_size, sell_cap)

        if buy_cap > 0 and bid_size > 0:
            orders.append(Order(product, bid_quote, bid_size))
        if sell_cap > 0 and ask_size > 0:
            orders.append(Order(product, ask_quote, -ask_size))

        return orders

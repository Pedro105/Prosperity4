from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Any
import json
import math

# BEGIN PARAMS
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
        "entry_edge_floor": 2.5,
        "entry_edge_mult": 1.10,
        "quote_edge_floor": 1.0,
        "quote_edge_mult": 0.60,
        "take_inventory_skew": 0.07,
        "reservation_inventory_skew": 0.15,
        "flatten_threshold": 50,
        "base_size": 22,
        "min_size": 6,
        "size_position_step": 5,
        "baseline_micro_weight": 0.55,
        "baseline_fast_ema_weight": 0.25,
        "baseline_slow_ema_weight": 0.20,
        "linreg_fair_value_mult": 0.7,
        "prediction_scale": 1.10,
        "prediction_clip": 10,
        "signal_floor": 0.12,
        "confidence_quote_tighten": 0.18,
        "size_tilt_strength": 2,
        "size_tilt_threshold": 0.75,
    },
}
# END PARAMS


# BEGIN LINEAR MODEL
LINEAR_MODEL = {
    "feature_names": [
        "spread",
        "micro_minus_mid",
        "imbalance_l1",
        "imbalance_l3",
        "ret_1",
        "ret_3",
        "ret_6",
        "ema_gap",
        "mean_reversion_gap",
        "avg_move"
    ],
    "means": {
        "spread": 13.020064044831383,
        "micro_minus_mid": -0.0043615097338658695,
        "imbalance_l1": -0.0002881039540950706,
        "imbalance_l3": -6.268413209614535e-05,
        "ret_1": -0.0021765235664965477,
        "ret_3": -0.006604623236265386,
        "ret_6": -0.013109176423496448,
        "ema_gap": -0.020727826794256013,
        "mean_reversion_gap": 0.008660876460879789,
        "avg_move": 1.5084262108475932
    },
    "stds": {
        "spread": 1.7548667709940504,
        "micro_minus_mid": 0.3230919850628444,
        "imbalance_l1": 0.08542710094354214,
        "imbalance_l3": 0.0270734373931867,
        "ret_1": 1.341280000630484,
        "ret_3": 1.5293333694465763,
        "ret_6": 1.7664919020530114,
        "ema_gap": 0.9384403118565846,
        "mean_reversion_gap": 1.0019997260256968,
        "avg_move": 0.05511465974572046
    },
    "coefficients": {
        "spread": -0.03137141267023864,
        "micro_minus_mid": -0.21342461466523852,
        "imbalance_l1": 0.4665051480370547,
        "imbalance_l3": -0.6695238469265514,
        "ret_1": -0.026600857531249963,
        "ret_3": 0.01488873123043606,
        "ret_6": -0.004398198397041495,
        "ema_gap": 0.015064334653624317,
        "mean_reversion_gap": 0.10502344020404596,
        "avg_move": 0.0031974208196821024
    },
    "intercept": -0.004486473865040763,
    "target": "mean_future_mid_delta",
    "target_definition": "mean(mid[t+1:t+h]) - mid[t]",
    "horizon": 3,
    "alpha": 10.0
}
# END LINEAR MODEL


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
        conversions = 0
        return result, conversions, trader_data

    # ----------------------------
    # State helpers
    # ----------------------------
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

    # ----------------------------
    # Generic helpers
    # ----------------------------
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

    def clamp(self, value: int, low: int, high: int) -> int:
        return max(low, min(high, value))

    def clamp_float(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def take_liquidity_buy(
        self,
        product: str,
        orders: List[Order],
        order_depth: OrderDepth,
        max_price: int,
        position: int,
        limit: int,
    ) -> int:
        if not order_depth.sell_orders:
            return position

        for ask in sorted(order_depth.sell_orders.keys()):
            if ask > max_price:
                break
            available = -order_depth.sell_orders[ask]
            qty = min(available, limit - position)
            if qty > 0:
                orders.append(Order(product, ask, qty))
                position += qty
        return position

    def take_liquidity_sell(
        self,
        product: str,
        orders: List[Order],
        order_depth: OrderDepth,
        min_price: int,
        position: int,
        limit: int,
    ) -> int:
        if not order_depth.buy_orders:
            return position

        for bid in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid < min_price:
                break
            available = order_depth.buy_orders[bid]
            qty = min(available, limit + position)
            if qty > 0:
                orders.append(Order(product, bid, -qty))
                position -= qty
        return position

    # ----------------------------
    # EMERALDS: stable around 10000
    # ----------------------------
    def trade_emeralds(self, order_depth: OrderDepth, position: int) -> List[Order]:
        product = "EMERALDS"
        limit = self.POSITION_LIMITS[product]
        fair_value = self.cfg(product, "fair_value")
        orders: List[Order] = []
        take_edge = self.cfg(product, "take_edge")

        best_bid, best_ask = self.best_bid_ask(order_depth)

        position = self.take_liquidity_buy(
            product, orders, order_depth,
            max_price=fair_value - take_edge,
            position=position, limit=limit
        )
        position = self.take_liquidity_sell(
            product, orders, order_depth,
            min_price=fair_value + take_edge,
            position=position, limit=limit
        )

        flatten_threshold = self.cfg(product, "flatten_threshold")
        if position > flatten_threshold:
            position = self.take_liquidity_sell(
                product, orders, order_depth,
                min_price=fair_value,
                position=position, limit=limit
            )
        elif position < -flatten_threshold:
            position = self.take_liquidity_buy(
                product, orders, order_depth,
                max_price=fair_value,
                position=position, limit=limit
            )

        spread_half = self.cfg(product, "spread_half")
        reservation = fair_value - self.cfg(product, "inventory_skew") * position

        raw_bid = math.floor(reservation - spread_half)
        raw_ask = math.ceil(reservation + spread_half)

        if best_bid is not None:
            bid_quote = min(raw_bid, best_bid + 1)
        else:
            bid_quote = raw_bid

        if best_ask is not None:
            ask_quote = max(raw_ask, best_ask - 1)
        else:
            ask_quote = raw_ask

        if bid_quote >= ask_quote:
            bid_quote = ask_quote - 1

        bid_quote = min(bid_quote, fair_value)
        ask_quote = max(ask_quote, fair_value)

        buy_cap = limit - position
        sell_cap = limit + position

        base_size = self.cfg(product, "base_size")
        min_size = self.cfg(product, "min_size")
        size_position_step = self.cfg(product, "size_position_step")
        bid_size = self.clamp(base_size - max(position, 0) // size_position_step, min_size, buy_cap)
        ask_size = self.clamp(base_size + min(position, 0) // size_position_step, min_size, sell_cap)

        if buy_cap > 0 and bid_size > 0:
            orders.append(Order(product, bid_quote, bid_size))

        if sell_cap > 0 and ask_size > 0:
            orders.append(Order(product, ask_quote, -ask_size))

        return orders

    # ----------------------------
    # TOMATOES: baseline market making + linear regression fair-value overlay
    # ----------------------------
    def build_tomatoes_feature_table(
        self,
        order_depth: OrderDepth,
        history: List[float],
        mid: float,
        micro: float,
        ema: float,
        fast_ema: float,
        slow_ema: float,
        avg_move: float,
    ) -> Dict[str, float]:
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

        ret_1 = 0.0 if len(history) < 2 else mid - history[-2]
        ret_3 = 0.0 if len(history) < 4 else mid - history[-4]
        ret_6 = 0.0 if len(history) < 7 else mid - history[-7]

        return {
            "spread": spread,
            "micro_minus_mid": micro - mid,
            "imbalance_l1": imbalance(float(bid_l1), float(ask_l1)),
            "imbalance_l3": imbalance(float(bid_l3), float(ask_l3)),
            "ret_1": ret_1,
            "ret_3": ret_3,
            "ret_6": ret_6,
            "ema_gap": fast_ema - slow_ema,
            "mean_reversion_gap": ema - mid,
            "avg_move": avg_move,
        }

    def predict_tomatoes_delta(self, features: Dict[str, float]) -> float:
        raw_prediction = LINEAR_MODEL.get("intercept", 0.0)
        for feature_name in LINEAR_MODEL.get("feature_names", []):
            mean = LINEAR_MODEL["means"].get(feature_name, 0.0)
            std = LINEAR_MODEL["stds"].get(feature_name, 1.0) or 1.0
            coefficient = LINEAR_MODEL["coefficients"].get(feature_name, 0.0)
            z_score = (features.get(feature_name, 0.0) - mean) / std
            raw_prediction += coefficient * z_score

        scaled_prediction = self.cfg("TOMATOES", "prediction_scale") * raw_prediction
        max_shift = self.cfg("TOMATOES", "prediction_clip")
        return self.clamp_float(scaled_prediction, -max_shift, max_shift)

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

        prev_ema = memory["ema"]["TOMATOES"]
        alpha = self.cfg(product, "ema_alpha")
        ema = mid if prev_ema is None else (alpha * mid + (1 - alpha) * prev_ema)
        memory["ema"]["TOMATOES"] = ema

        prev_fast_ema = memory["fast_ema"]["TOMATOES"]
        fast_alpha = self.cfg(product, "fast_ema_alpha")
        fast_ema = mid if prev_fast_ema is None else (fast_alpha * mid + (1 - fast_alpha) * prev_fast_ema)
        memory["fast_ema"]["TOMATOES"] = fast_ema

        prev_slow_ema = memory["slow_ema"]["TOMATOES"]
        slow_alpha = self.cfg(product, "slow_ema_alpha")
        slow_ema = mid if prev_slow_ema is None else (slow_alpha * mid + (1 - slow_alpha) * prev_slow_ema)
        memory["slow_ema"]["TOMATOES"] = slow_ema

        if len(history) >= 2:
            recent_moves = [abs(history[i] - history[i - 1]) for i in range(1, len(history))]
            avg_move_lookback = self.cfg(product, "avg_move_lookback")
            avg_move = sum(recent_moves[-avg_move_lookback:]) / min(len(recent_moves), avg_move_lookback)
        else:
            avg_move = 2.0
        avg_move = max(self.cfg(product, "min_avg_move"), avg_move)

        features = self.build_tomatoes_feature_table(
            order_depth=order_depth,
            history=history,
            mid=mid,
            micro=micro,
            ema=ema,
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            avg_move=avg_move,
        )
        predicted_delta = self.predict_tomatoes_delta(features)
        if abs(predicted_delta) < self.cfg(product, "signal_floor"):
            predicted_delta = 0.0

        base_fair_value = (
            self.cfg(product, "baseline_micro_weight") * micro
            + self.cfg(product, "baseline_fast_ema_weight") * fast_ema
            + self.cfg(product, "baseline_slow_ema_weight") * slow_ema
        )
        fair_value = base_fair_value + self.cfg(product, "linreg_fair_value_mult") * predicted_delta

        entry_edge = max(
            self.cfg(product, "entry_edge_floor"),
            self.cfg(product, "entry_edge_mult") * avg_move,
        )
        quote_edge = max(
            self.cfg(product, "quote_edge_floor"),
            self.cfg(product, "quote_edge_mult") * avg_move,
        )

        confidence = min(1.0, abs(predicted_delta) / max(avg_move, 1.0))
        quote_edge *= max(
            0.80,
            1.0 - self.cfg(product, "confidence_quote_tighten") * confidence,
        )

        take_inventory_skew = self.cfg(product, "take_inventory_skew")
        take_buy_price = math.floor(fair_value - entry_edge - take_inventory_skew * position)
        take_sell_price = math.ceil(fair_value + entry_edge - take_inventory_skew * position)

        position = self.take_liquidity_buy(
            product, orders, order_depth,
            max_price=take_buy_price,
            position=position, limit=limit
        )
        position = self.take_liquidity_sell(
            product, orders, order_depth,
            min_price=take_sell_price,
            position=position, limit=limit
        )

        flatten_threshold = self.cfg(product, "flatten_threshold")
        if position > flatten_threshold:
            position = self.take_liquidity_sell(
                product, orders, order_depth,
                min_price=math.floor(fair_value),
                position=position, limit=limit
            )
        elif position < -flatten_threshold:
            position = self.take_liquidity_buy(
                product, orders, order_depth,
                max_price=math.ceil(fair_value),
                position=position, limit=limit
            )

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
        size_position_step = self.cfg(product, "size_position_step")
        bid_size = self.clamp(base_size - max(position, 0) // size_position_step, min_size, buy_cap)
        ask_size = self.clamp(base_size + min(position, 0) // size_position_step, min_size, sell_cap)

        size_tilt = 0
        if abs(predicted_delta) >= self.cfg(product, "size_tilt_threshold"):
            size_tilt = self.cfg(product, "size_tilt_strength")

        if predicted_delta > 0:
            bid_size = self.clamp(bid_size + size_tilt, min_size, buy_cap)
            ask_size = self.clamp(ask_size - size_tilt, min_size, sell_cap)
        elif predicted_delta < 0:
            ask_size = self.clamp(ask_size + size_tilt, min_size, sell_cap)
            bid_size = self.clamp(bid_size - size_tilt, min_size, buy_cap)

        if buy_cap > 0 and bid_size > 0:
            orders.append(Order(product, bid_quote, bid_size))

        if sell_cap > 0 and ask_size > 0:
            orders.append(Order(product, ask_quote, -ask_size))

        return orders

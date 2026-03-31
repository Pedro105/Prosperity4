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
        "fast_ema_alpha": 0.35,
        "slow_ema_alpha": 0.08,
        "avg_move_lookback": 16,
        "min_avg_move": 1.5,
        "entry_edge_floor": 2.5,
        "entry_edge_mult": 1.10,
        "quote_edge_floor": 1.0,
        "quote_edge_mult": 0.60,
        "take_inventory_skew": 0.07,
        "reservation_inventory_skew": 0.15,
        "flatten_threshold": 50,
        "prediction_bias_mult": 2.20,
        "confidence_quote_tighten": 0.30,
        "base_size": 22,
        "min_size": 6,
        "size_position_step": 5,
    },
}


# BEGIN LOGREG MODEL
MODEL = {
    "feature_names": [
        "spread",
        "micro_minus_mid",
        "imbalance_l1",
        "imbalance_l3",
        "ret_1",
        "ret_3",
        "ret_6",
        "ema_gap",
        "avg_move"
    ],
    "means": {
        "spread": 13.020266212970377,
        "micro_minus_mid": -0.004261866169988154,
        "imbalance_l1": -0.00026311277154443953,
        "imbalance_l3": -6.55228601710602e-05,
        "ret_1": -0.002101681345076061,
        "ret_3": -0.006505204163330665,
        "ret_6": -0.012860288230584467,
        "ema_gap": -0.021620761565554186,
        "avg_move": 0.7896066853482786
    },
    "stds": {
        "spread": 1.7545815404569105,
        "micro_minus_mid": 0.3227995811671308,
        "imbalance_l1": 0.08535820479562127,
        "imbalance_l3": 0.027071824329463447,
        "ret_1": 1.3409134677741232,
        "ret_3": 1.5292712598395302,
        "ret_6": 1.7662917550267654,
        "ema_gap": 0.93876275794921,
        "avg_move": 0.35567738818329037
    },
    "coefficients": {
        "spread": 0.07382200937664385,
        "micro_minus_mid": 0.2164079387587734,
        "imbalance_l1": 0.19577639120641,
        "imbalance_l3": -0.823912001077901,
        "ret_1": -0.017537806202237605,
        "ret_3": -0.005861668210961928,
        "ret_6": -0.04301269416861886,
        "ema_gap": -0.04352308512117975,
        "avg_move": -0.009486936397555327
    },
    "intercept": -0.4411656656923342,
    "horizon": 3
}
# END LOGREG MODEL


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    DEFAULT_STATE = {
        "mid_history": {"TOMATOES": []},
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

        return result, 0, json.dumps(memory, separators=(",", ":"))

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

    def sigmoid(self, x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def build_tomatoes_feature_table(
        self,
        order_depth: OrderDepth,
        history: List[float],
        mid: float,
        micro: float,
        fast_ema: float,
        slow_ema: float,
    ) -> Dict[str, float]:
        """
        This feature table mirrors the training pipeline.

        Why these features:
        - spread: wider markets reduce fill quality and can change quoting style
        - micro_minus_mid: level-1 order book pressure is often predictive short-term
        - imbalance_l1 / imbalance_l3: captures whether bid or ask depth dominates
        - ret_1 / ret_3 / ret_6: short-horizon momentum and reversion structure
        - ema_gap: smooth trend signal without using a heavy model
        - avg_move: local volatility used to scale how strongly we trust the signal
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

        ret_1 = 0.0 if len(history) < 2 else mid - history[-2]
        ret_3 = 0.0 if len(history) < 4 else mid - history[-4]
        ret_6 = 0.0 if len(history) < 7 else mid - history[-7]

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
            "ret_1": ret_1,
            "ret_3": ret_3,
            "ret_6": ret_6,
            "ema_gap": fast_ema - slow_ema,
            "avg_move": max(self.cfg("TOMATOES", "min_avg_move"), avg_move),
        }

    def predict_tomatoes_up_probability(self, features: Dict[str, float]) -> float:
        score = MODEL.get("intercept", 0.0)
        for feature_name in MODEL.get("feature_names", []):
            mean = MODEL["means"].get(feature_name, 0.0)
            std = MODEL["stds"].get(feature_name, 1.0) or 1.0
            coefficient = MODEL["coefficients"].get(feature_name, 0.0)
            z = (features.get(feature_name, 0.0) - mean) / std
            score += coefficient * z
        return self.sigmoid(score)

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

        fast_alpha = self.cfg(product, "fast_ema_alpha")
        prev_fast = memory["fast_ema"]["TOMATOES"]
        fast_ema = mid if prev_fast is None else fast_alpha * mid + (1 - fast_alpha) * prev_fast
        memory["fast_ema"]["TOMATOES"] = fast_ema

        slow_alpha = self.cfg(product, "slow_ema_alpha")
        prev_slow = memory["slow_ema"]["TOMATOES"]
        slow_ema = mid if prev_slow is None else slow_alpha * mid + (1 - slow_alpha) * prev_slow
        memory["slow_ema"]["TOMATOES"] = slow_ema

        features = self.build_tomatoes_feature_table(order_depth, history, mid, micro, fast_ema, slow_ema)
        prob_up = self.predict_tomatoes_up_probability(features)
        signal = 2.0 * (prob_up - 0.5)  # maps probability into [-1, 1]
        confidence = abs(signal)

        base_fair_value = 0.55 * micro + 0.25 * fast_ema + 0.20 * slow_ema
        fair_value = base_fair_value + signal * self.cfg(product, "prediction_bias_mult") * features["avg_move"]

        entry_edge = max(
            self.cfg(product, "entry_edge_floor"),
            self.cfg(product, "entry_edge_mult") * features["avg_move"],
        )
        quote_edge = max(
            self.cfg(product, "quote_edge_floor"),
            self.cfg(product, "quote_edge_mult") * features["avg_move"],
        )

        # Higher-confidence predictions justify quoting a bit tighter because
        # we want more fills in the direction the model prefers.
        quote_edge *= max(0.75, 1.0 - self.cfg(product, "confidence_quote_tighten") * confidence)

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
        bid_size = self.clamp(base_size - max(position, 0) // step, min_size, buy_cap)
        ask_size = self.clamp(base_size + min(position, 0) // step, min_size, sell_cap)

        # The model output also tilts sizes: when probability of an up move is
        # high we are slightly happier to own inventory and less eager to sell.
        if signal > 0:
            bid_size = self.clamp(bid_size + 2, min_size, buy_cap)
            ask_size = self.clamp(ask_size - 2, min_size, sell_cap)
        elif signal < 0:
            ask_size = self.clamp(ask_size + 2, min_size, sell_cap)
            bid_size = self.clamp(bid_size - 2, min_size, buy_cap)

        if buy_cap > 0 and bid_size > 0:
            orders.append(Order(product, bid_quote, bid_size))
        if sell_cap > 0 and ask_size > 0:
            orders.append(Order(product, ask_quote, -ask_size))

        return orders

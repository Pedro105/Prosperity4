
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Any
import json
import math


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    DEFAULT_STATE = {
        "mid_history": {
            "TOMATOES": []
        },
        "ema": {
            "TOMATOES": None
        }
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
            if "mid_history" not in loaded:
                loaded["mid_history"] = {"TOMATOES": []}
            if "ema" not in loaded:
                loaded["ema"] = {"TOMATOES": None}
            if "TOMATOES" not in loaded["mid_history"]:
                loaded["mid_history"]["TOMATOES"] = []
            if "TOMATOES" not in loaded["ema"]:
                loaded["ema"]["TOMATOES"] = None
            return loaded
        except Exception:
            return json.loads(json.dumps(self.DEFAULT_STATE))

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

        # Gives more weight to the side with less resting liquidity
        return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

    def clamp(self, value: int, low: int, high: int) -> int:
        return max(low, min(high, value))

    def take_liquidity_buy(self, product: str, orders: List[Order], order_depth: OrderDepth,
                           max_price: int, position: int, limit: int) -> int:
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

    def take_liquidity_sell(self, product: str, orders: List[Order], order_depth: OrderDepth,
                            min_price: int, position: int, limit: int) -> int:
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
        fair_value = 10000
        orders: List[Order] = []

        best_bid, best_ask = self.best_bid_ask(order_depth)

        # 1) Aggressively take clear mispricings
        position = self.take_liquidity_buy(
            product, orders, order_depth,
            max_price=fair_value - 1,
            position=position, limit=limit
        )
        position = self.take_liquidity_sell(
            product, orders, order_depth,
            min_price=fair_value + 1,
            position=position, limit=limit
        )

        # 2) If inventory is stretched, flatten closer to fair value
        if position > 50:
            position = self.take_liquidity_sell(
                product, orders, order_depth,
                min_price=fair_value,
                position=position, limit=limit
            )
        elif position < -50:
            position = self.take_liquidity_buy(
                product, orders, order_depth,
                max_price=fair_value,
                position=position, limit=limit
            )

        # 3) Passive market making with inventory-aware reservation price
        spread_half = 2
        reservation = fair_value - 0.08 * position

        raw_bid = math.floor(reservation - spread_half)
        raw_ask = math.ceil(reservation + spread_half)

        # Improve current book by 1 tick when possible, but don't cross
        if best_bid is not None:
            bid_quote = min(raw_bid, best_bid + 1)
        else:
            bid_quote = raw_bid

        if best_ask is not None:
            ask_quote = max(raw_ask, best_ask - 1)
        else:
            ask_quote = raw_ask

        # Maintain a sane two-sided market
        if bid_quote >= ask_quote:
            bid_quote = ask_quote - 1

        bid_quote = min(bid_quote, fair_value)
        ask_quote = max(ask_quote, fair_value)

        buy_cap = limit - position
        sell_cap = limit + position

        # More size when near flat, less size when inventory is already large
        base_size = 24
        bid_size = self.clamp(base_size - max(position, 0) // 4, 8, buy_cap)
        ask_size = self.clamp(base_size + min(position, 0) // 4, 8, sell_cap)

        if buy_cap > 0 and bid_size > 0:
            orders.append(Order(product, bid_quote, bid_size))

        if sell_cap > 0 and ask_size > 0:
            orders.append(Order(product, ask_quote, -ask_size))

        return orders

    # ----------------------------
    # TOMATOES: adaptive mean reversion
    # ----------------------------
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
        if len(history) > 40:
            history[:] = history[-40:]

        prev_ema = memory["ema"]["TOMATOES"]
        alpha = 0.20
        ema = mid if prev_ema is None else (alpha * mid + (1 - alpha) * prev_ema)
        memory["ema"]["TOMATOES"] = ema

        # Use recent absolute move as a lightweight volatility estimate
        if len(history) >= 2:
            recent_moves = [abs(history[i] - history[i - 1]) for i in range(1, len(history))]
            avg_move = sum(recent_moves[-12:]) / min(len(recent_moves), 12)
        else:
            avg_move = 2.0

        avg_move = max(1.5, avg_move)
        fair_value = 0.60 * micro + 0.40 * ema

        # Inventory-aware thresholds:
        # smaller threshold when flat, larger when already leaning the same way
        entry_edge = max(2.5, 1.25 * avg_move)
        take_buy_price = math.floor(fair_value - entry_edge - 0.05 * position)
        take_sell_price = math.ceil(fair_value + entry_edge - 0.05 * position)

        # 1) Take obvious mispricings
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

        # 2) If inventory is extreme, prioritize reducing it
        if position > 55:
            reduce_price = math.floor(fair_value)
            position = self.take_liquidity_sell(
                product, orders, order_depth,
                min_price=reduce_price,
                position=position, limit=limit
            )
        elif position < -55:
            reduce_price = math.ceil(fair_value)
            position = self.take_liquidity_buy(
                product, orders, order_depth,
                max_price=reduce_price,
                position=position, limit=limit
            )

        # 3) Passive quotes around reservation price
        reservation = fair_value - 0.12 * position
        quote_edge = max(1.0, 0.75 * avg_move)

        raw_bid = math.floor(reservation - quote_edge)
        raw_ask = math.ceil(reservation + quote_edge)

        bid_quote = min(raw_bid, best_bid + 1)
        ask_quote = max(raw_ask, best_ask - 1)

        if bid_quote >= ask_quote:
            bid_quote = ask_quote - 1

        buy_cap = limit - position
        sell_cap = limit + position

        # Dynamic size: larger when edge is wider and inventory is manageable
        base_size = 22 if avg_move < 3 else 18
        bid_size = self.clamp(base_size - max(position, 0) // 5, 6, buy_cap)
        ask_size = self.clamp(base_size + min(position, 0) // 5, 6, sell_cap)

        if buy_cap > 0 and bid_size > 0:
            orders.append(Order(product, bid_quote, bid_size))

        if sell_cap > 0 and ask_size > 0:
            orders.append(Order(product, ask_quote, -ask_size))

        return orders

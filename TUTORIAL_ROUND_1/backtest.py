"""
Backtesting engine for Prosperity 4 trading challenge.

Reads the provided prices_round_X_day_Y.csv files and replays them as
TradingState objects fed into the Trader.run() method, then tracks PnL.

This is a local approximation of the IMC/Prosperity environment:
- it calls the submitted Trader class directly
- it preserves traderData across iterations
- it resets inventory/state per price file (each sample day is independent)
- it rejects over-limit order batches before matching

Usage:
    python backtest.py
    python backtest.py trader123
"""

import importlib
import sys
from collections import defaultdict

import pandas as pd
from datamodel import (
    Listing, OrderDepth, Trade, TradingState, Observation, Order
)

# ── Configuration ────────────────────────────────────────────────────────────

PRICE_FILES = [
    "prices_round_0_day_-2.csv",
    "prices_round_0_day_-1.csv",
]

DEFAULT_POSITION_LIMITS: dict[str, int] = {
    "EMERALDS": 20,
    "TOMATOES": 20,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_order_depth(row: pd.Series) -> OrderDepth:
    """Parse up to 3 bid/ask levels from a price-file row."""
    od = OrderDepth()
    for i in range(1, 4):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and bv > 0:
            od.buy_orders[int(bp)] = int(bv)

        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(ap) and pd.notna(av) and av > 0:
            od.sell_orders[int(ap)] = -int(av)   # sell quantities stored negative

    return od


def get_position_limits(trader: object) -> dict[str, int]:
    trader_limits = getattr(trader, "POSITION_LIMITS", None)
    if isinstance(trader_limits, dict) and trader_limits:
        return dict(trader_limits)
    return dict(DEFAULT_POSITION_LIMITS)


def filter_valid_orders(
    product: str,
    orders: list[Order],
    current_position: int,
    position_limits: dict[str, int],
) -> list[Order]:
    """
    Mirror exchange-side position checks:
    if the aggregate buy or sell quantity for a product would breach the limit
    assuming full execution, that side's orders are rejected.
    """
    limit = position_limits.get(product, DEFAULT_POSITION_LIMITS.get(product, 20))

    buy_total = sum(order.quantity for order in orders if order.quantity > 0)
    sell_total = sum(-order.quantity for order in orders if order.quantity < 0)

    buy_valid = current_position + buy_total <= limit
    sell_valid = current_position - sell_total >= -limit

    return [
        order for order in orders
        if (order.quantity > 0 and buy_valid) or (order.quantity < 0 and sell_valid)
    ]


def match_orders(
    product: str,
    orders: list[Order],
    order_depth: OrderDepth,
    position: dict[str, int],
    own_trades_out: dict[str, list[Trade]],
    timestamp: int,
    cash: dict[str, float],
    position_limits: dict[str, int],
) -> None:
    """
    Simulate exchange matching for one product.
    Fills buy orders against ask levels and sell orders against bid levels.
    Respects position limits.
    """
    pos = position.get(product, 0)
    limit = position_limits.get(product, DEFAULT_POSITION_LIMITS.get(product, 20))

    buy_orders  = sorted([o for o in orders if o.quantity > 0], key=lambda o: -o.price)
    sell_orders = sorted([o for o in orders if o.quantity < 0], key=lambda o:  o.price)

    # Available liquidity copies (so we don't mutate the original)
    asks = dict(order_depth.sell_orders)   # price -> negative qty
    bids = dict(order_depth.buy_orders)    # price -> positive qty

    for order in buy_orders:
        remaining = order.quantity
        for ask_price in sorted(asks.keys()):
            if ask_price > order.price:
                break
            available = -asks[ask_price]          # positive units available
            max_buyable = limit - pos
            fill = min(remaining, available, max_buyable)
            if fill <= 0:
                break
            own_trades_out[product].append(
                Trade(product, ask_price, fill, buyer="SUBMISSION", seller="", timestamp=timestamp)
            )
            cash[product] = cash.get(product, 0) - fill * ask_price
            pos += fill
            remaining -= fill
            asks[ask_price] += fill
            if asks[ask_price] == 0:
                del asks[ask_price]
            if remaining == 0:
                break

    for order in sell_orders:
        remaining = abs(order.quantity)
        for bid_price in sorted(bids.keys(), reverse=True):
            if bid_price < order.price:
                break
            available = bids[bid_price]
            max_sellable = pos + limit
            fill = min(remaining, available, max_sellable)
            if fill <= 0:
                break
            own_trades_out[product].append(
                Trade(product, bid_price, fill, buyer="", seller="SUBMISSION", timestamp=timestamp)
            )
            cash[product] = cash.get(product, 0) + fill * bid_price
            pos -= fill
            remaining -= fill
            bids[bid_price] -= fill
            if bids[bid_price] == 0:
                del bids[bid_price]
            if remaining == 0:
                break

    position[product] = pos


# ── Main backtester ───────────────────────────────────────────────────────────

def run_backtest(trader_module: str = "trader123") -> None:
    # Dynamic import so you can modify trader.py without restarting
    if trader_module in sys.modules:
        module = importlib.reload(sys.modules[trader_module])
    else:
        module = importlib.import_module(trader_module)
    trader_class = module.Trader

    for price_file in PRICE_FILES:
        trader = trader_class()
        position_limits = get_position_limits(trader)
        position: dict[str, int] = {}
        cash: dict[str, float] = defaultdict(float)
        trader_data = ""
        total_pnl_history: list[tuple[int, float]] = []
        product_pnl_history: dict[str, list[tuple[int, float]]] = defaultdict(list)

        print(f"\n{'='*60}")
        print(f"  Backtesting on: {price_file} with {trader_module}.py")
        print(f"{'='*60}")

        try:
            prices_df = pd.read_csv(price_file, sep=";")
        except FileNotFoundError:
            print(f"  [WARN] File not found, skipping: {price_file}")
            continue

        products = prices_df["product"].unique().tolist()

        listings = {
            p: Listing(symbol=p, product=p, denomination="XIRECS")
            for p in products
        }
        observations = Observation(
            plainValueObservations={},
            conversionObservations={},
        )

        prev_own_trades: dict[str, list[Trade]] = {p: [] for p in products}

        for timestamp, group in prices_df.groupby("timestamp"):
            order_depths: dict[str, OrderDepth] = {}
            for _, row in group.iterrows():
                order_depths[row["product"]] = build_order_depth(row)

            state = TradingState(
                traderData=trader_data,
                timestamp=int(timestamp),
                listings=listings,
                order_depths=order_depths,
                own_trades=prev_own_trades,
                market_trades={p: [] for p in products},
                position=dict(position),
                observations=observations,
            )

            try:
                result, conversions, trader_data = trader.run(state)
            except Exception as exc:
                print(f"  [ERROR] trader.run raised at t={timestamp}: {exc}")
                continue

            new_own_trades: dict[str, list[Trade]] = {p: [] for p in products}

            for product, orders in result.items():
                if product not in order_depths:
                    continue
                valid_orders = filter_valid_orders(
                    product=product,
                    orders=orders,
                    current_position=position.get(product, 0),
                    position_limits=position_limits,
                )
                match_orders(
                    product=product,
                    orders=valid_orders,
                    order_depth=order_depths[product],
                    position=position,
                    own_trades_out=new_own_trades,
                    timestamp=int(timestamp),
                    cash=cash,
                    position_limits=position_limits,
                )

            prev_own_trades = new_own_trades

            # Mark-to-mid PnL snapshot (total and per-product)
            mid_prices = {
                row["product"]: row["mid_price"]
                for _, row in group.iterrows()
            }
            total_pnl = 0.0
            for p in products:
                p_pnl = cash[p] + position.get(p, 0) * mid_prices.get(p, 0.0)
                product_pnl_history[p].append((int(timestamp), p_pnl))
                total_pnl += p_pnl
            total_pnl_history.append((int(timestamp), total_pnl))

        # Per-day summary
        day_cash = sum(cash.values())
        day_pos_val = sum(
            position.get(p, 0) * prices_df[prices_df["product"] == p]["mid_price"].iloc[-1]
            for p in products
            if not prices_df[prices_df["product"] == p].empty
        )
        print(f"\n  Cash P&L       : {day_cash:>12.2f} XIRECS")
        print(f"  Open position  : {dict(position)}")
        print(f"  Position value : {day_pos_val:>12.2f} XIRECS")
        print(f"  Total mark-PnL : {day_cash + day_pos_val:>12.2f} XIRECS")
        print(f"  Position limits: {position_limits}")

        # Optional: plot per-product + total PnL curves
        try:
            import matplotlib.pyplot as plt

            COLORS = {
                "EMERALDS": "#2ecc71",
                "TOMATOES": "#e74c3c",
                "TOTAL": "#3498db",
            }

            fig, ax = plt.subplots(figsize=(14, 6))

            for product, history in product_pnl_history.items():
                if history:
                    ts, pnls = zip(*history)
                    color = COLORS.get(product, None)
                    ax.plot(ts, pnls, linewidth=1, label=product, color=color, alpha=0.85)

            if total_pnl_history:
                ts, pnls = zip(*total_pnl_history)
                ax.plot(
                    ts,
                    pnls,
                    linewidth=1.5,
                    label="TOTAL",
                    color=COLORS["TOTAL"],
                    linestyle="--",
                    alpha=0.9,
                )

            ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
            ax.set_title(f"Mark-to-Mid PnL — {price_file} ({trader_module}.py)")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("PnL (XIRECS)")
            ax.legend()
            fig.tight_layout()
            chart_name = f"pnl_curve_{price_file.replace('.csv', '')}.png"
            fig.savefig(chart_name, dpi=150)
            plt.close(fig)
            print(f"  PnL chart saved to {chart_name}")
        except ImportError:
            pass

    print(f"\n{'='*60}")
    print("  Backtest complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_backtest(sys.argv[1] if len(sys.argv) > 1 else "trader123")

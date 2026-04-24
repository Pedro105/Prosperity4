#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "ROUND_2"
OUTPUT_DIR = REPO_ROOT / "Data_Exploration" / "round2_plots"

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
PRODUCTS = [PEPPER, OSMIUM]
DAYS = [-1, 0, 1]


def load_prices() -> pd.DataFrame:
    frames = []
    for day in DAYS:
        df = pd.read_csv(DATA_DIR / f"prices_round_2_day_{day}.csv", sep=";")
        frames.append(df)
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    prices["spread"] = prices["ask_price_1"] - prices["bid_price_1"]
    prices["microprice"] = (
        prices["ask_price_1"] * prices["bid_volume_1"] + prices["bid_price_1"] * prices["ask_volume_1"]
    ) / (prices["bid_volume_1"] + prices["ask_volume_1"])
    prices["imbalance_l1"] = (
        (prices["bid_volume_1"] - prices["ask_volume_1"]) / (prices["bid_volume_1"] + prices["ask_volume_1"])
    )
    prices["mid_return_1"] = prices.groupby(["product", "day"])["mid_price"].diff()
    prices["mid_return_5"] = prices.groupby(["product", "day"])["mid_price"].diff(5)
    prices["future_return_1"] = prices.groupby(["product", "day"])["mid_price"].shift(-1) - prices["mid_price"]
    prices["future_return_5"] = prices.groupby(["product", "day"])["mid_price"].shift(-5) - prices["mid_price"]
    prices["realized_vol_50"] = (
        prices.groupby(["product", "day"])["mid_return_1"].transform(lambda s: s.rolling(50).std())
    )
    return prices


def load_trades() -> pd.DataFrame:
    frames = []
    for day in DAYS:
        df = pd.read_csv(DATA_DIR / f"trades_round_2_day_{day}.csv", sep=";")
        df["day"] = day
        frames.append(df)
    trades = pd.concat(frames, ignore_index=True)
    trades = trades.sort_values(["symbol", "day", "timestamp"]).reset_index(drop=True)
    return trades


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def plot_midprice_and_trend(prices: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, len(DAYS), figsize=(16, 7), sharex=False)
    for row_idx, product in enumerate(PRODUCTS):
        product_df = prices[prices["product"] == product]
        for col_idx, day in enumerate(DAYS):
            ax = axes[row_idx, col_idx]
            day_df = product_df[product_df["day"] == day]
            ax.plot(day_df["timestamp"], day_df["mid_price"], linewidth=1.1, color="steelblue", label="mid")
            if product == PEPPER:
                x = day_df["timestamp"].to_numpy()
                y = day_df["mid_price"].to_numpy()
                slope, intercept = np.polyfit(x, y, 1)
                ax.plot(x, slope * x + intercept, linestyle="--", color="darkorange", label="linear fit")
            ax.set_title(f"{product} | day {day}")
            ax.set_xlabel("timestamp")
            ax.set_ylabel("mid")
            if row_idx == 0 and col_idx == 0:
                ax.legend()
    fig.suptitle("Mid-price paths and PEPPER trend fit")
    savefig("midprice_paths_and_trend.png")


def plot_spread_regimes(prices: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, product in zip(axes[0], PRODUCTS):
        product_df = prices[prices["product"] == product]
        sns.histplot(product_df["spread"].dropna(), bins=40, kde=True, ax=ax, color="slateblue")
        ax.set_title(f"{product} spread distribution")
        ax.set_xlabel("spread")
    for ax, product in zip(axes[1], PRODUCTS):
        product_df = prices[prices["product"] == product]
        sns.lineplot(data=product_df, x="timestamp", y="spread", hue="day", palette="viridis", ax=ax, legend=True)
        ax.set_title(f"{product} spread over time")
        ax.set_xlabel("timestamp")
        ax.set_ylabel("spread")
    fig.suptitle("Spread regimes")
    savefig("spread_regimes.png")


def plot_imbalance_signal(prices: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, product in zip(axes, PRODUCTS):
        product_df = prices[prices["product"] == product].dropna(subset=["imbalance_l1", "future_return_5"]).copy()
        product_df["imbalance_bucket"] = pd.cut(product_df["imbalance_l1"], bins=np.linspace(-1, 1, 11))
        bucketed = (
            product_df.groupby("imbalance_bucket", observed=False)["future_return_5"]
            .agg(["mean", "count"])
            .reset_index()
        )
        centers = [interval.mid for interval in bucketed["imbalance_bucket"]]
        ax.plot(centers, bucketed["mean"], marker="o", color="firebrick")
        ax.axhline(0, linestyle=":", color="black", linewidth=0.8)
        ax.set_title(f"{product} imbalance vs 5-tick future return")
        ax.set_xlabel("L1 imbalance bucket")
        ax.set_ylabel("avg future mid return")
    fig.suptitle("Order book imbalance signal")
    savefig("imbalance_signal.png")


def plot_microprice_edge(prices: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, product in zip(axes, PRODUCTS):
        product_df = prices[prices["product"] == product].dropna(subset=["microprice", "future_return_1"]).copy()
        product_df["micro_edge"] = product_df["microprice"] - product_df["mid_price"]
        product_df["edge_bucket"] = pd.cut(product_df["micro_edge"], bins=15)
        bucketed = (
            product_df.groupby("edge_bucket", observed=False)["future_return_1"]
            .mean()
            .reset_index()
        )
        centers = [interval.mid for interval in bucketed["edge_bucket"]]
        ax.plot(centers, bucketed["future_return_1"], marker="o", color="darkgreen")
        ax.axhline(0, linestyle=":", color="black", linewidth=0.8)
        ax.set_title(f"{product} microprice edge vs next-tick return")
        ax.set_xlabel("microprice - mid")
        ax.set_ylabel("avg next-tick mid return")
    fig.suptitle("Microprice lead signal")
    savefig("microprice_edge_signal.png")


def plot_realized_volatility(prices: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, product in zip(axes, PRODUCTS):
        product_df = prices[prices["product"] == product]
        sns.lineplot(data=product_df, x="timestamp", y="realized_vol_50", hue="day", palette="magma", ax=ax)
        ax.set_title(f"{product} rolling realized volatility (50 ticks)")
        ax.set_ylabel("std of 1-tick mid return")
    axes[-1].set_xlabel("timestamp")
    fig.suptitle("Intraday volatility regimes")
    savefig("realized_volatility_regimes.png")


def plot_displayed_depth(prices: pd.DataFrame) -> None:
    records = []
    for product in PRODUCTS:
        product_df = prices[prices["product"] == product]
        for level in [1, 2, 3]:
            for side in ["bid", "ask"]:
                volume = product_df[f"{side}_volume_{level}"].abs()
                records.append(
                    {
                        "product": product,
                        "level": f"L{level}",
                        "side": side.upper(),
                        "mean_volume": float(volume.mean()),
                    }
                )
    depth_df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, product in zip(axes, PRODUCTS):
        subset = depth_df[depth_df["product"] == product]
        sns.barplot(data=subset, x="level", y="mean_volume", hue="side", ax=ax, palette="Set2")
        ax.set_title(f"{product} average displayed depth by level")
        ax.set_ylabel("mean absolute volume")
    fig.suptitle("Displayed liquidity profile")
    savefig("displayed_depth_profile.png")


def plot_trade_activity(trades: pd.DataFrame) -> None:
    trade_bins = []
    for product in PRODUCTS:
        product_df = trades[trades["symbol"] == product].copy()
        product_df["time_bin"] = (product_df["timestamp"] // 5000) * 5000
        summary = (
            product_df.groupby(["day", "time_bin"])
            .agg(trade_count=("quantity", "size"), total_qty=("quantity", "sum"), vwap=("price", "mean"))
            .reset_index()
        )
        summary["product"] = product
        trade_bins.append(summary)
    summary_df = pd.concat(trade_bins, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    for row_idx, product in enumerate(PRODUCTS):
        subset = summary_df[summary_df["product"] == product]
        sns.lineplot(data=subset, x="time_bin", y="trade_count", hue="day", ax=axes[row_idx, 0], palette="tab10")
        axes[row_idx, 0].set_title(f"{product} trade count per 5k ticks")
        axes[row_idx, 0].set_ylabel("trade count")
        sns.lineplot(data=subset, x="time_bin", y="total_qty", hue="day", ax=axes[row_idx, 1], palette="tab10")
        axes[row_idx, 1].set_title(f"{product} traded quantity per 5k ticks")
        axes[row_idx, 1].set_ylabel("total quantity")
    axes[1, 0].set_xlabel("timestamp bucket")
    axes[1, 1].set_xlabel("timestamp bucket")
    fig.suptitle("Trade activity intensity")
    savefig("trade_activity_intensity.png")


def plot_osmium_maker_diagnostics(prices: pd.DataFrame) -> None:
    osmium = prices[prices["product"] == OSMIUM].copy()
    maker_records = []
    for level in [1, 2, 3]:
        detect = osmium[f"bid_volume_{level}"].abs().between(20, 30) | osmium[f"ask_volume_{level}"].abs().between(20, 30)
        maker_records.append({"level": f"L{level}", "pct_detected": float(detect.mean() * 100)})
    maker_df = pd.DataFrame(maker_records)

    candidates = []
    for _, row in osmium.iterrows():
        mids = []
        for level in [1, 2, 3]:
            bid_q = abs(row.get(f"bid_volume_{level}", np.nan))
            ask_q = abs(row.get(f"ask_volume_{level}", np.nan))
            bid_p = row.get(f"bid_price_{level}", np.nan)
            ask_p = row.get(f"ask_price_{level}", np.nan)
            if pd.notna(bid_p) and pd.notna(ask_p) and (
                (pd.notna(bid_q) and 20 <= bid_q <= 30) or (pd.notna(ask_q) and 20 <= ask_q <= 30)
            ):
                mids.append((bid_p + ask_p) / 2.0)
        candidates.append(np.nan if not mids else np.mean(mids) - row["mid_price"])
    osmium["maker_mid_delta"] = candidates

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=maker_df, x="level", y="pct_detected", ax=axes[0], color="cornflowerblue")
    axes[0].set_title("OSMIUM maker-like size detection rate")
    axes[0].set_ylabel("% of ticks")
    sns.histplot(osmium["maker_mid_delta"].dropna(), bins=40, kde=True, ax=axes[1], color="teal")
    axes[1].axvline(0, linestyle=":", color="black", linewidth=0.8)
    axes[1].set_title("OSMIUM maker-mid minus raw mid")
    axes[1].set_xlabel("maker-mid delta")
    fig.suptitle("OSMIUM maker-layer diagnostics")
    savefig("osmium_maker_diagnostics.png")


def write_summary(prices: pd.DataFrame, trades: pd.DataFrame) -> None:
    lines = []
    lines.append("Round 2 visualization summary")
    lines.append("=" * 32)
    lines.append("")
    for product in PRODUCTS:
        product_df = prices[prices["product"] == product]
        product_trades = trades[trades["symbol"] == product]
        lines.append(product)
        lines.append(f"  mean spread: {product_df['spread'].mean():.3f}")
        lines.append(f"  median spread: {product_df['spread'].median():.3f}")
        lines.append(f"  mean abs 1-tick return: {product_df['mid_return_1'].abs().mean():.4f}")
        lines.append(f"  trade count: {len(product_trades)}")
        lines.append(f"  total traded quantity: {product_trades['quantity'].sum():.0f}")
        lines.append("")
    (OUTPUT_DIR / "plot_summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    prices = load_prices()
    trades = load_trades()

    plot_midprice_and_trend(prices)
    plot_spread_regimes(prices)
    plot_imbalance_signal(prices)
    plot_microprice_edge(prices)
    plot_realized_volatility(prices)
    plot_displayed_depth(prices)
    plot_trade_activity(trades)
    plot_osmium_maker_diagnostics(prices)
    write_summary(prices, trades)

    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

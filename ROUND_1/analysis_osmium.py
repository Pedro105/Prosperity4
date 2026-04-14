#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

PRODUCT = "ASH_COATED_OSMIUM"
REFERENCE_BETA = -0.229
ROLLING_WINDOW = 200
PLOT_DPI = 150

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots" / "analysis"
DATA_FILES = [
    ("Day -2", BASE_DIR / "prices_round_1_day_-2.csv"),
    ("Day -1", BASE_DIR / "prices_round_1_day_-1.csv"),
    ("Day 0", BASE_DIR / "prices_round_1_day_0.csv"),
]


def plot_stem(label: str) -> str:
    return {"Day -2": "day-2", "Day -1": "day-1", "Day 0": "day0", "Combined": "combined"}[label]


def load_product_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    out = df[df["product"] == PRODUCT].copy()
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out["mid_price"] = pd.to_numeric(out["mid_price"], errors="coerce")
    out = out.dropna(subset=["timestamp", "mid_price"])
    out = out[out["mid_price"] > 0]
    return out.sort_values(["timestamp"]).reset_index(drop=True)


def interpret_adf(p_value: float) -> str:
    return "STATIONARY — mean reversion is valid" if p_value < 0.05 else "NON-STATIONARY — random walk, mean reversion is risky"


def interpret_acf(lag1: float) -> str:
    if lag1 <= -0.2:
        return "Strong mean reversion signal"
    if -0.05 <= lag1 <= 0.05:
        return "Random walk — market making only"
    if lag1 > 0.05:
        return "Momentum — mean reversion will lose money"
    return "Weak negative autocorrelation — some mean reversion tendency"


def regime_shares(rolling: pd.Series) -> tuple[float, float, float]:
    valid = rolling.dropna()
    if valid.empty:
        return float("nan"), float("nan"), float("nan")
    mr = 100 * (valid < -0.05).mean()
    mom = 100 * (valid > 0.05).mean()
    neutral = 100 - mr - mom
    return mr, mom, neutral


def print_dataset_header(label: str, n: int) -> None:
    print("\n" + "=" * 72)
    print(f"DATASET: {label} ({PRODUCT}, n_mid_prices={n})")
    print("=" * 72)


def analyze_dataset(label: str, df: pd.DataFrame) -> dict[str, Any]:
    stem = plot_stem(label)
    prices = df["mid_price"].to_numpy(dtype=float)
    returns = (
        pd.Series(prices)
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    print_dataset_header(label, len(prices))

    adf_stat, adf_p = adfuller(prices, autolag="AIC")[:2]
    print("\n--- Step 1: Augmented Dickey-Fuller (levels: mid_price) ---")
    print(f"ADF Statistic: {adf_stat:.6f}")
    print(f"p-value: {adf_p:.6f}")
    print(f"Interpretation: {interpret_adf(adf_p)}")

    lag1 = float(returns.autocorr(lag=1))
    print("\n--- Step 2: Autocorrelation of returns ---")
    print(f"Lag-1 autocorrelation: {lag1:.6f}")
    print(f"Interpretation: {interpret_acf(lag1)}")

    fig_acf, ax_acf = plt.subplots(figsize=(9, 4))
    plot_acf(returns, lags=20, alpha=0.05, ax=ax_acf, title=f"{PRODUCT} — ACF of returns ({label})")
    ax_acf.set_xlabel("Lag")
    ax_acf.set_ylabel("Autocorrelation")
    fig_acf.tight_layout()
    acf_path = PLOTS_DIR / f"acf_{stem}.png"
    fig_acf.savefig(acf_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig_acf)
    print(f"Saved: {acf_path}")

    x = returns.iloc[:-1].to_numpy().reshape(-1, 1)
    y = returns.iloc[1:].to_numpy()
    model = LinearRegression().fit(x, y)
    beta = float(model.coef_[0])
    r_squared = float(model.score(x, y))
    print("\n--- Step 3: Fit reversion beta via linear regression ---")
    print(f"Your reversion beta: {beta:.6f}")
    print(f"R² of the model: {r_squared:.6f}")
    print(f"Reference beta (starfruit.py): {REFERENCE_BETA:.6f}")
    if beta < REFERENCE_BETA:
        print("Interpretation: stronger (more negative) than the reference beta.")
    elif beta > REFERENCE_BETA:
        print("Interpretation: weaker (less negative) than the reference beta.")
    else:
        print("Interpretation: comparable to the reference beta.")
    print("Interpretation:", "model has meaningful predictive power." if r_squared >= 0.05 else "model has limited predictive power.")

    fig_scatter, ax_scatter = plt.subplots(figsize=(7, 7))
    ax_scatter.scatter(x[:, 0], y, s=8, alpha=0.35, label="Observed")
    xs = np.linspace(x.min(), x.max(), 200)
    ax_scatter.plot(xs, model.predict(xs.reshape(-1, 1)), color="crimson", lw=2, label=f"OLS line (β={beta:.4f})")
    ax_scatter.axhline(0, color="grey", lw=0.8)
    ax_scatter.axvline(0, color="grey", lw=0.8)
    ax_scatter.set_xlabel("return[t]")
    ax_scatter.set_ylabel("return[t+1]")
    ax_scatter.set_title(f"{PRODUCT} — return[t+1] vs return[t] ({label})")
    ax_scatter.legend()
    ax_scatter.grid(True, alpha=0.3)
    scatter_path = PLOTS_DIR / f"beta_scatter_{stem}.png"
    fig_scatter.savefig(scatter_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig_scatter)
    print(f"Saved: {scatter_path}")

    rolling = returns.rolling(window=ROLLING_WINDOW).apply(lambda s: s.autocorr(lag=1), raw=False)
    mr_pct, mom_pct, neutral_pct = regime_shares(rolling)
    print("\n--- Step 4: Rolling autocorrelation (regime detection) ---")
    print(f"% of timestamps in mean reversion regime: {mr_pct:.2f}%")
    print(f"% of timestamps in momentum regime: {mom_pct:.2f}%")
    print(f"% of timestamps in neutral/random regime: {neutral_pct:.2f}%")

    idx = np.arange(len(returns))
    fig_roll, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(idx, rolling.to_numpy(), color="steelblue", lw=1.2, label="Rolling lag-1 autocorr")
    ax1.axhline(0, color="black", ls="--", lw=1)
    ax1.axhspan(-1.0, -0.05, color="green", alpha=0.12, label="Mean Reversion Regime")
    ax1.axhspan(0.05, 1.0, color="red", alpha=0.12, label="Momentum Regime")
    ax1.set_xlabel("Observation index")
    ax1.set_ylabel("Rolling lag-1 autocorr")
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{PRODUCT} — rolling autocorrelation & mid_price ({label})")

    ax2 = ax1.twinx()
    ax2.plot(idx, df["mid_price"].iloc[1 : len(returns) + 1].to_numpy(), color="lightgrey", lw=0.8, label="mid_price")
    ax2.set_ylabel("mid_price", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    roll_path = PLOTS_DIR / f"rolling_autocorr_{stem}.png"
    fig_roll.savefig(roll_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig_roll)
    print(f"Saved: {roll_path}")

    return {
        "label": label,
        "adf_p": float(adf_p),
        "stationary": adf_p < 0.05,
        "lag1_ac": lag1,
        "beta": beta,
        "r2": r_squared,
        "pct_mr": mr_pct,
        "pct_mom": mom_pct,
        "pct_neutral": neutral_pct,
    }


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 88)
    print("SUMMARY TABLE")
    print("=" * 88)
    print(f"{'Dataset':<12}{'ADF p-value':>14}{'Stationary?':>14}{'Lag-1 AC':>12}{'Beta':>12}{'R²':>10}{'% MR Regime':>14}")
    print("-" * 88)
    for row in rows:
        print(
            f"{row['label']:<12}"
            f"{row['adf_p']:>14.6f}"
            f"{('Yes' if row['stationary'] else 'No'):>14}"
            f"{row['lag1_ac']:>12.4f}"
            f"{row['beta']:>12.4f}"
            f"{row['r2']:>10.4f}"
            f"{row['pct_mr']:>14.1f}"
        )


def print_recommendation(combined: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("RECOMMENDATION")
    print("=" * 88)
    if combined["stationary"]:
        print(f"- {PRODUCT} levels look stationary on the combined sample, so a mean-reversion view is more defensible.")
    else:
        print(f"- {PRODUCT} levels do not look stationary on the combined sample, so treating it as anchored to one fixed mean is risky.")
    if combined["lag1_ac"] < -0.05:
        print(f"- Returns show negative lag-1 autocorrelation, so short-horizon mean reversion is plausible.")
        print(f"- A reasonable starting beta for a trader is {combined['beta']:.4f}.")
    elif combined["lag1_ac"] > 0.05:
        print(f"- Returns show positive lag-1 autocorrelation, so momentum is more plausible than mean reversion.")
    else:
        print(f"- Returns are near-random at lag 1, so market making is safer than directional prediction.")
    if combined["pct_mr"] > 60:
        print("- The rolling autocorrelation spends most of its time in the mean-reversion regime, so regime switching is probably not necessary.")
    else:
        print("- The rolling autocorrelation changes enough that a regime-aware approach may help.")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = [(label, load_product_prices(path)) for label, path in DATA_FILES]
    combined = pd.concat([df.assign(day_order=i) for i, (_, df) in enumerate(datasets)], ignore_index=True)
    combined = combined.sort_values(["day_order", "timestamp"]).drop(columns=["day_order"]).reset_index(drop=True)

    results = [analyze_dataset(label, df) for label, df in datasets]
    results.append(analyze_dataset("Combined", combined))
    print_summary(results)
    print_recommendation(results[-1])


if __name__ == "__main__":
    main()

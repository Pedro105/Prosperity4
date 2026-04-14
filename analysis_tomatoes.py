#!/usr/bin/env python3
"""
Mean-reversion diagnostics for TOMATOES (IMC Prosperity) — ADF, return ACF,
AR(1) beta on returns, rolling lag-1 autocorrelation with regime shading.

Run: python analysis_tomatoes.py
Optional: python analysis_tomatoes.py /path/to/folder_with_csvs
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

PRODUCT = "TOMATOES"
REFERENCE_BETA = -0.229  # starfruit.py reversion_beta
ROLLING_WINDOW = 200
PLOT_DPI = 150

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "TUTORIAL_ROUND_1"
PLOTS_DIR = BASE_DIR / "plots" / "analysis"

DATA_FILES: list[tuple[str, str]] = [
    ("Day -2", "prices_round_0_day_-2.csv"),
    ("Day -1", "prices_round_0_day_-1.csv"),
    ("Day 0", "prices_round_0_day_0.csv"),
]


def safe_plot_stem(label: str) -> str:
    return {
        "Day -2": "day-2",
        "Day -1": "day-1",
        "Day 0": "day0",
        "Combined": "combined",
    }.get(label, label.lower().replace(" ", "_"))


def load_tomatoes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    t = df[df["product"] == PRODUCT].copy()
    t["timestamp"] = pd.to_numeric(t["timestamp"], errors="coerce")
    t["mid_price"] = pd.to_numeric(t["mid_price"], errors="coerce")
    t = t.dropna(subset=["timestamp", "mid_price"])
    return t.sort_values(["timestamp", "mid_price"]).drop_duplicates(subset=["timestamp"], keep="last")


def interpret_adf(p: float) -> str:
    if p < 0.05:
        return "STATIONARY — mean reversion is valid"
    return "NON-STATIONARY — random walk, mean reversion is risky"


def interpret_lag1(ac1: float) -> str:
    if ac1 < -0.05:
        if ac1 <= -0.2:
            return "Strong mean reversion signal"
        return "Negative serial correlation in returns (mean reversion tendency)"
    if ac1 > 0.05:
        return "Momentum — mean reversion will lose money"
    return "Random walk — market making only"


def run_adf(prices: np.ndarray) -> tuple[float, float]:
    clean = prices[~np.isnan(prices)]
    if len(clean) < 10:
        return float("nan"), float("nan")
    res = adfuller(clean, autolag="AIC")
    return float(res[0]), float(res[1])


def regime_fractions(rolling: pd.Series) -> tuple[float, float, float]:
    valid = rolling.dropna()
    if valid.empty:
        return float("nan"), float("nan"), float("nan")
    n = len(valid)
    mr = (valid < -0.05).sum() / n * 100
    mom = (valid > 0.05).sum() / n * 100
    neutral = 100.0 - mr - mom
    return mr, mom, neutral


def fit_ar1_beta(returns: pd.Series) -> tuple[float, float]:
    X = returns.iloc[:-1].values.reshape(-1, 1)
    y = returns.iloc[1:].values
    if len(X) < 2 or not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        return float("nan"), float("nan")
    model = LinearRegression().fit(X, y)
    return float(model.coef_[0]), float(model.score(X, y))


def analyze_one(label: str, tom: pd.DataFrame, plots_dir: Path) -> dict[str, Any]:
    stem = safe_plot_stem(label)
    prices = tom["mid_price"].astype(float).values

    print("\n" + "=" * 72)
    print(f"DATASET: {label}  (n_mid_prices={len(prices)})")
    print("=" * 72)

    adf_stat, adf_p = run_adf(prices)
    print("\n--- Step 1: Augmented Dickey-Fuller (levels: mid_price) ---")
    print(f"ADF Statistic: {adf_stat:.6f}")
    print(f"p-value: {adf_p:.6f}")
    print(f"Interpretation: {interpret_adf(adf_p)}")

    s = pd.Series(prices)
    returns = s.pct_change().dropna()
    if len(returns) < 3:
        print("ERROR: Not enough observations for returns analysis.")
        return {
            "label": label,
            "adf_p": adf_p,
            "stationary": adf_p < 0.05 if not np.isnan(adf_p) else False,
            "lag1_ac": float("nan"),
            "beta": float("nan"),
            "r2": float("nan"),
            "pct_mr": float("nan"),
            "pct_mom": float("nan"),
            "pct_neutral": float("nan"),
        }

    lag1 = float(returns.autocorr(lag=1))
    print("\n--- Step 2: Autocorrelation of returns ---")
    print(f"Lag-1 autocorrelation: {lag1:.6f}")
    print(f"Interpretation: {interpret_lag1(lag1)}")

    fig_acf, ax_acf = plt.subplots(figsize=(9, 4))
    plot_acf(returns, lags=20, alpha=0.05, ax=ax_acf, title=f"TOMATOES — ACF of returns ({label})")
    ax_acf.set_xlabel("Lag")
    ax_acf.set_ylabel("Autocorrelation")
    plt.tight_layout()
    acf_path = plots_dir / f"acf_{stem}.png"
    fig_acf.savefig(acf_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig_acf)
    print(f"Saved: {acf_path}")

    beta, r2 = fit_ar1_beta(returns)
    print("\n--- Step 3: AR(1) on returns (t+1 ~ t) ---")
    if lag1 < 0:
        print(f"Your reversion beta: {beta:.6f}")
        print(f"R² of the model: {r2:.6f}")
        print(f"Reference beta (starfruit.py): {REFERENCE_BETA:.6f}")
        if np.isfinite(beta):
            if beta < REFERENCE_BETA:
                print("Compared to reference: stronger (more negative) reversion in returns.")
            elif beta > REFERENCE_BETA:
                print("Compared to reference: weaker (less negative) reversion in returns.")
            else:
                print("Compared to reference: comparable.")
        if np.isfinite(r2) and r2 < 0.01:
            print("R² is tiny — limited linear predictive power from one lag of returns.")
    else:
        print("Lag-1 autocorrelation is not negative — mean reversion on returns is not supported by lag-1 AC.")
        print(f"(Still reporting OLS beta for reference: {beta:.6f}, R²: {r2:.6f})")
        print(f"Reference beta (starfruit.py): {REFERENCE_BETA:.6f}")

    fig_sc, ax_sc = plt.subplots(figsize=(7, 7))
    x_r = returns.iloc[:-1].values
    y_r = returns.iloc[1:].values
    ax_sc.scatter(x_r, y_r, s=8, alpha=0.35, label="Observed")
    if np.isfinite(beta):
        xs = np.linspace(np.nanmin(x_r), np.nanmax(x_r), 100)
        ax_sc.plot(xs, beta * xs, color="crimson", lw=2, label=f"OLS line (β={beta:.4f})")
    ax_sc.axhline(0, color="grey", lw=0.8)
    ax_sc.axvline(0, color="grey", lw=0.8)
    ax_sc.set_xlabel("return[t]")
    ax_sc.set_ylabel("return[t+1]")
    ax_sc.set_title(f"TOMATOES — return[t+1] vs return[t] ({label})")
    ax_sc.legend()
    ax_sc.grid(True, alpha=0.3)
    sc_path = plots_dir / f"beta_scatter_{stem}.png"
    fig_sc.savefig(sc_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig_sc)
    print(f"Saved: {sc_path}")

    rolling_autocorr = returns.rolling(window=ROLLING_WINDOW).apply(
        lambda x: x.autocorr(lag=1) if len(x.dropna()) >= 2 else np.nan,
        raw=False,
    )
    mr_pct, mom_pct, neu_pct = regime_fractions(rolling_autocorr)

    print("\n--- Step 4: Rolling lag-1 autocorrelation ---")
    print(f"Window: {ROLLING_WINDOW} observations")
    print(f"% valid rolling obs in mean-reversion regime (ρ₁ < -0.05): {mr_pct:.2f}%")
    print(f"% in momentum regime (ρ₁ > 0.05): {mom_pct:.2f}%")
    print(f"% neutral (-0.05 ≤ ρ₁ ≤ 0.05): {neu_pct:.2f}%")

    x_idx = np.arange(len(returns))
    fig_r, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(x_idx, rolling_autocorr.values, color="steelblue", lw=1.2, label="Rolling lag-1 autocorr")
    ax1.axhline(0, color="black", ls="--", lw=1)
    ax1.axhspan(-1.0, -0.05, color="green", alpha=0.12, label="Mean reversion regime")
    ax1.axhspan(0.05, 1.0, color="red", alpha=0.12, label="Momentum regime")
    ax1.set_ylabel("Rolling autocorr (lag 1)")
    ax1.set_xlabel("Observation index (returns)")
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"TOMATOES — rolling autocorrelation & mid_price ({label})")

    ax2 = ax1.twinx()
    price_aligned = tom["mid_price"].astype(float).iloc[1 : len(returns) + 1].values
    ax2.plot(x_idx, price_aligned, color="grey", alpha=0.45, lw=0.8, label="mid_price")
    ax2.set_ylabel("mid_price", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    roll_path = plots_dir / f"rolling_autocorr_{stem}.png"
    fig_r.savefig(roll_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig_r)
    print(f"Saved: {roll_path}")

    return {
        "label": label,
        "adf_p": adf_p,
        "stationary": adf_p < 0.05 if not np.isnan(adf_p) else False,
        "lag1_ac": lag1,
        "beta": beta,
        "r2": r2,
        "pct_mr": mr_pct,
        "pct_mom": mom_pct,
        "pct_neutral": neu_pct,
    }


def build_combined(datasets: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    parts = []
    for label, tom in datasets:
        d = tom.copy()
        d["_order"] = {"Day -2": 0, "Day -1": 1, "Day 0": 2}.get(label, 3)
        parts.append(d)
    comb = pd.concat(parts, ignore_index=True)
    return comb.sort_values(["_order", "timestamp"]).drop(columns=["_order"]).reset_index(drop=True)


def print_summary_table(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    hdr = (
        f"{'Dataset':<12} {'ADF p-value':>14} {'Stationary?':>12} {'Lag-1 AC':>12} "
        f"{'Beta':>10} {'R²':>10} {'% MR Regime':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        lab = r["label"][:12]
        p = r["adf_p"]
        stat = "Yes" if r.get("stationary") else "No"
        if isinstance(p, float) and np.isnan(p):
            ps, stat = "N/A", "N/A"
        else:
            ps = f"{p:.6f}"
        l1 = r["lag1_ac"]
        l1s = f"{l1:.4f}" if isinstance(l1, float) and np.isfinite(l1) else "N/A"
        b = r["beta"]
        bs = f"{b:.4f}" if isinstance(b, float) and np.isfinite(b) else "N/A"
        r2v = r["r2"]
        r2s = f"{r2v:.4f}" if isinstance(r2v, float) and np.isfinite(r2v) else "N/A"
        mr = r["pct_mr"]
        mrs = f"{mr:.1f}" if isinstance(mr, float) and np.isfinite(mr) else "N/A"
        print(f"{lab:<12} {ps:>14} {stat:>12} {l1s:>12} {bs:>10} {r2s:>10} {mrs:>12}")


def final_recommendation(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("RECOMMENDATION (plain English)")
    print("=" * 72)
    combined = next((r for r in rows if r["label"] == "Combined"), None)
    if not combined:
        return
    adf_ok = combined.get("stationary")
    l1 = combined.get("lag1_ac")
    mr_share = combined.get("pct_mr")
    beta_c = combined.get("beta")

    if adf_ok is True:
        print("- ADF on combined mid_price suggests stationarity of levels (check per-day robustness).")
    elif adf_ok is False:
        print("- ADF on combined mid_price does not reject a unit root; prefer inference on returns.")

    if isinstance(l1, float) and np.isfinite(l1):
        if l1 < -0.05:
            print("- Negative lag-1 return autocorrelation supports short-horizon mean-reversion-style fading.")
        elif l1 > 0.05:
            print("- Positive lag-1 return autocorrelation suggests momentum risk for naive MR.")
        else:
            print("- Lag-1 return autocorrelation near zero — random-walk-like; MM-style logic may dominate.")

    if isinstance(mr_share, float) and np.isfinite(mr_share):
        print(
            f"- Rolling ρ₁ spends ~{mr_share:.1f}% of valid windows below -0.05 "
            "(mean-reversion band); compare to momentum % in the log above."
        )

    if isinstance(beta_c, float) and np.isfinite(beta_c) and isinstance(l1, float) and l1 < 0:
        print(
            f"- For `trader_rolling_avg.py`, a data-driven TOMATOES beta candidate from this OLS is ~{beta_c:.4f} "
            "(re-fit when Day 0 CSV is available)."
        )
    else:
        print("- Use conservative MR beta or rely on MM until lag-1 AC is clearly negative on your full sample.")


def main() -> int:
    data_dir = Path(DEFAULT_DATA_DIR)
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1]).resolve()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Plots directory: {PLOTS_DIR}")
    print(f"Product: {PRODUCT}")

    loaded: list[tuple[str, pd.DataFrame]] = []
    missing: list[str] = []

    for label, fname in DATA_FILES:
        path = data_dir / fname
        if not path.is_file():
            missing.append(f"{label}: {path}")
            continue
        loaded.append((label, load_tomatoes(path)))
        print(f"Loaded {label}: {path} ({len(loaded[-1][1])} rows)")

    if missing:
        print("\nWARNING — missing files:")
        for m in missing:
            print(f"  {m}")

    if not loaded:
        print("No price files found. Exiting.")
        return 1

    results: list[dict[str, Any]] = []
    for label, tom in loaded:
        results.append(analyze_one(label, tom, PLOTS_DIR))

    if len(loaded) >= 2:
        results.append(analyze_one("Combined", build_combined(loaded), PLOTS_DIR))

    labels_done = {r["label"] for r in results}
    for label, _ in DATA_FILES:
        if label not in labels_done:
            results.append(
                {
                    "label": label,
                    "adf_p": float("nan"),
                    "stationary": False,
                    "lag1_ac": float("nan"),
                    "beta": float("nan"),
                    "r2": float("nan"),
                    "pct_mr": float("nan"),
                    "pct_mom": float("nan"),
                    "pct_neutral": float("nan"),
                }
            )

    order = {"Day -2": 0, "Day -1": 1, "Day 0": 2, "Combined": 3}
    results.sort(key=lambda r: order.get(r["label"], 99))

    print_summary_table(results)
    final_recommendation(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_TRAIN = [WORKSPACE / "prices_round_0_day_-2.csv"]
DEFAULT_VALIDATE = [WORKSPACE / "prices_round_0_day_-1.csv"]
TRADER_FILE = WORKSPACE / "trader_linreg.py"
MODEL_JSON = WORKSPACE / "tomatoes_linreg_model.json"
BEGIN_MARKER = "# BEGIN LINEAR MODEL"
END_MARKER = "# END LINEAR MODEL"
FEATURE_NAMES = [
    "spread",
    "micro_minus_mid",
    "imbalance_l1",
    "imbalance_l3",
    "ret_1",
    "ret_3",
    "ret_6",
    "ema_gap",
    "mean_reversion_gap",
    "avg_move",
]
HORIZON_CANDIDATES = [2, 3, 4, 5]
ALPHA_CANDIDATES = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
MIN_AVG_MOVE = 1.5
AVG_MOVE_LOOKBACK = 16
EMA_ALPHA = 0.20
FAST_EMA_ALPHA = 0.35
SLOW_EMA_ALPHA = 0.08


def load_prices(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, sep=";")
        df = df[df["product"] == "TOMATOES"].copy()
        df["source_file"] = path.name
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["source_file", "timestamp"]).reset_index(drop=True)


def add_features(raw_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frames = []

    for _, group in raw_df.groupby("source_file", sort=False):
        out = group.sort_values("timestamp").copy()

        numeric_columns = [
            "bid_price_1",
            "ask_price_1",
            "bid_volume_1",
            "ask_volume_1",
            "mid_price",
        ]
        for level in range(1, 4):
            numeric_columns.extend(
                [
                    f"bid_price_{level}",
                    f"ask_price_{level}",
                    f"bid_volume_{level}",
                    f"ask_volume_{level}",
                ]
            )

        for column in numeric_columns:
            if column in out:
                out[column] = pd.to_numeric(out[column], errors="coerce")

        out["mid"] = out["mid_price"]
        out["spread"] = out["ask_price_1"] - out["bid_price_1"]

        bid_l1 = out["bid_volume_1"].fillna(0.0)
        ask_l1 = out["ask_volume_1"].fillna(0.0)
        out["microprice"] = (
            out["bid_price_1"] * ask_l1 + out["ask_price_1"] * bid_l1
        ) / (bid_l1 + ask_l1).replace(0, pd.NA)
        out["microprice"] = out["microprice"].fillna(out["mid"])
        out["micro_minus_mid"] = out["microprice"] - out["mid"]

        bid_l3 = sum(out.get(f"bid_volume_{level}", 0).fillna(0.0) for level in range(1, 4))
        ask_l3 = sum(out.get(f"ask_volume_{level}", 0).fillna(0.0) for level in range(1, 4))

        def imbalance(bid_volume: pd.Series, ask_volume: pd.Series) -> pd.Series:
            return (bid_volume - ask_volume) / (bid_volume + ask_volume).replace(0, pd.NA)

        out["imbalance_l1"] = imbalance(bid_l1, ask_l1)
        out["imbalance_l3"] = imbalance(bid_l3, ask_l3)

        out["ret_1"] = out["mid"].diff(1)
        out["ret_3"] = out["mid"].diff(3)
        out["ret_6"] = out["mid"].diff(6)

        ema = out["mid"].ewm(alpha=EMA_ALPHA, adjust=False).mean()
        fast_ema = out["mid"].ewm(alpha=FAST_EMA_ALPHA, adjust=False).mean()
        slow_ema = out["mid"].ewm(alpha=SLOW_EMA_ALPHA, adjust=False).mean()
        out["ema_gap"] = fast_ema - slow_ema
        out["mean_reversion_gap"] = ema - out["mid"]

        avg_move = out["mid"].diff().abs().rolling(AVG_MOVE_LOOKBACK).mean()
        avg_move = avg_move.fillna(2.0)
        out["avg_move"] = avg_move.clip(lower=MIN_AVG_MOVE)

        future_mid_mean = pd.concat(
            [out["mid"].shift(-step) for step in range(1, horizon + 1)],
            axis=1,
        ).mean(axis=1)
        out["future_mid_delta"] = future_mid_mean - out["mid"]

        out = out.dropna(subset=FEATURE_NAMES + ["future_mid_delta"]).reset_index(drop=True)
        frames.append(out)

    return pd.concat(frames, ignore_index=True)


def standardize(df: pd.DataFrame, means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    return (df[FEATURE_NAMES] - means) / stds


def evaluate_predictions(y_true: pd.Series, predictions: pd.Series) -> dict[str, float]:
    correlation = y_true.corr(predictions)
    sign_accuracy = ((y_true > 0) == (predictions > 0)).mean()
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
        "r2": float(r2_score(y_true, predictions)),
        "corr": 0.0 if pd.isna(correlation) else float(correlation),
        "sign_accuracy": float(sign_accuracy),
    }


def fit_model(train_df: pd.DataFrame, validate_df: pd.DataFrame, alpha: float) -> tuple[Ridge, dict, dict]:
    means = train_df[FEATURE_NAMES].mean()
    stds = train_df[FEATURE_NAMES].std().replace(0, 1.0).fillna(1.0)

    x_train = standardize(train_df, means, stds)
    y_train = train_df["future_mid_delta"]
    x_validate = standardize(validate_df, means, stds)
    y_validate = validate_df["future_mid_delta"]

    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)

    train_predictions = pd.Series(model.predict(x_train), index=train_df.index)
    validate_predictions = pd.Series(model.predict(x_validate), index=validate_df.index)

    metrics = {
        "train": evaluate_predictions(y_train, train_predictions),
        "validate": evaluate_predictions(y_validate, validate_predictions),
    }
    model_dict = {
        "feature_names": FEATURE_NAMES,
        "means": {name: float(means[name]) for name in FEATURE_NAMES},
        "stds": {name: float(stds[name]) for name in FEATURE_NAMES},
        "coefficients": {name: float(model.coef_[i]) for i, name in enumerate(FEATURE_NAMES)},
        "intercept": float(model.intercept_),
        "target": "mean_future_mid_delta",
        "target_definition": "mean(mid[t+1:t+h]) - mid[t]",
        "alpha": float(alpha),
    }
    return model, model_dict, metrics


def patch_trader_file(trader_path: Path, model_dict: dict) -> None:
    source = trader_path.read_text()
    start = source.index(BEGIN_MARKER)
    end = source.index(END_MARKER)
    replacement = (
        f"{BEGIN_MARKER}\n"
        f"LINEAR_MODEL = {json.dumps(model_dict, indent=4)}\n"
        f"{END_MARKER}"
    )
    updated = source[:start] + replacement + source[end + len(END_MARKER):]
    trader_path.write_text(updated)


def print_metrics(label: str, metrics: dict) -> None:
    print(
        f"{label}: "
        f"mae={metrics['mae']:.4f}, "
        f"rmse={metrics['rmse']:.4f}, "
        f"corr={metrics['corr']:.4f}, "
        f"sign_acc={metrics['sign_accuracy']:.4f}, "
        f"r2={metrics['r2']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a TOMATOES Ridge regression model and patch trader_linreg.py."
    )
    parser.add_argument(
        "--fit-all",
        action="store_true",
        help="Refit on train+validate after selecting the best horizon/alpha.",
    )
    args = parser.parse_args()

    train_raw = load_prices(DEFAULT_TRAIN)
    validate_raw = load_prices(DEFAULT_VALIDATE)

    best_run = None
    best_score = None

    for horizon in HORIZON_CANDIDATES:
        train_df = add_features(train_raw, horizon)
        validate_df = add_features(validate_raw, horizon)

        for alpha in ALPHA_CANDIDATES:
            _, model_dict, metrics = fit_model(train_df, validate_df, alpha)
            validate_metrics = metrics["validate"]
            score = (
                validate_metrics["corr"],
                -validate_metrics["mae"],
                validate_metrics["sign_accuracy"],
            )
            if best_score is None or score > best_score:
                best_score = score
                best_run = {
                    "horizon": horizon,
                    "alpha": alpha,
                    "train_df": train_df,
                    "validate_df": validate_df,
                    "model_dict": model_dict,
                    "metrics": metrics,
                }

    if best_run is None:
        raise RuntimeError("No valid training run found.")

    horizon = best_run["horizon"]
    alpha = best_run["alpha"]
    model_dict = best_run["model_dict"]
    model_dict["horizon"] = int(horizon)

    print(f"Selected horizon={horizon}, alpha={alpha}")
    print_metrics("train", best_run["metrics"]["train"])
    print_metrics("validate", best_run["metrics"]["validate"])

    if args.fit_all:
        combined_raw = pd.concat([train_raw, validate_raw], ignore_index=True)
        combined_df = add_features(combined_raw, horizon)
        means = combined_df[FEATURE_NAMES].mean()
        stds = combined_df[FEATURE_NAMES].std().replace(0, 1.0).fillna(1.0)
        x_all = standardize(combined_df, means, stds)
        y_all = combined_df["future_mid_delta"]

        model = Ridge(alpha=alpha)
        model.fit(x_all, y_all)

        all_predictions = pd.Series(model.predict(x_all), index=combined_df.index)
        print_metrics("refit_all", evaluate_predictions(y_all, all_predictions))

        model_dict = {
            "feature_names": FEATURE_NAMES,
            "means": {name: float(means[name]) for name in FEATURE_NAMES},
            "stds": {name: float(stds[name]) for name in FEATURE_NAMES},
            "coefficients": {name: float(model.coef_[i]) for i, name in enumerate(FEATURE_NAMES)},
            "intercept": float(model.intercept_),
            "target": "mean_future_mid_delta",
            "target_definition": "mean(mid[t+1:t+h]) - mid[t]",
            "horizon": int(horizon),
            "alpha": float(alpha),
        }

    MODEL_JSON.write_text(json.dumps(model_dict, indent=4))
    patch_trader_file(TRADER_FILE, model_dict)
    print(f"Wrote model to {MODEL_JSON}")
    print(f"Patched {TRADER_FILE}")


if __name__ == "__main__":
    main()

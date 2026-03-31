import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


WORKSPACE = Path("/Users/pedropinto/Downloads/TUTORIAL_ROUND_1")
DEFAULT_TRAIN = [WORKSPACE / "prices_round_0_day_-2.csv"]
DEFAULT_VALIDATE = [WORKSPACE / "prices_round_0_day_-1.csv"]
TRADER_FILE = WORKSPACE / "trader_logreg.py"
MODEL_JSON = WORKSPACE / "tomatoes_logreg_model.json"
BEGIN_MARKER = "# BEGIN LOGREG MODEL"
END_MARKER = "# END LOGREG MODEL"
FEATURE_NAMES = [
    "spread",
    "micro_minus_mid",
    "imbalance_l1",
    "imbalance_l3",
    "ret_1",
    "ret_3",
    "ret_6",
    "ema_gap",
    "avg_move",
]


def load_prices(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, sep=";")
        df = df[df["product"] == "TOMATOES"].copy()
        df["source_file"] = path.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["source_file", "timestamp"]).reset_index(drop=True)
    return out


def add_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df.copy()

    for col in ["bid_price_1", "ask_price_1", "bid_volume_1", "ask_volume_1"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["spread"] = out["ask_price_1"] - out["bid_price_1"]
    out["mid"] = out["mid_price"]
    out["microprice"] = (
        out["bid_price_1"] * out["ask_volume_1"] + out["ask_price_1"] * out["bid_volume_1"]
    ) / (out["bid_volume_1"] + out["ask_volume_1"]).replace(0, pd.NA)
    out["microprice"] = out["microprice"].fillna(out["mid"])
    out["micro_minus_mid"] = out["microprice"] - out["mid"]

    bid_l1 = out["bid_volume_1"].fillna(0.0)
    ask_l1 = out["ask_volume_1"].fillna(0.0)
    out["imbalance_l1"] = (bid_l1 - ask_l1) / (bid_l1 + ask_l1).replace(0, pd.NA)

    bid_l3 = sum(pd.to_numeric(out.get(f"bid_volume_{i}", 0), errors="coerce").fillna(0.0) for i in range(1, 4))
    ask_l3 = sum(pd.to_numeric(out.get(f"ask_volume_{i}", 0), errors="coerce").fillna(0.0) for i in range(1, 4))
    out["imbalance_l3"] = (bid_l3 - ask_l3) / (bid_l3 + ask_l3).replace(0, pd.NA)

    out["ret_1"] = out["mid"].diff(1)
    out["ret_3"] = out["mid"].diff(3)
    out["ret_6"] = out["mid"].diff(6)

    fast_ema = out["mid"].ewm(alpha=0.35, adjust=False).mean()
    slow_ema = out["mid"].ewm(alpha=0.08, adjust=False).mean()
    out["ema_gap"] = fast_ema - slow_ema
    out["avg_move"] = out["mid"].diff().abs().rolling(16).mean()

    out["future_return"] = out["mid"].shift(-horizon) - out["mid"]
    out["target_up"] = (out["future_return"] > 0).astype(int)

    out = out.dropna(subset=FEATURE_NAMES + ["target_up"]).reset_index(drop=True)
    return out


def evaluate(model: LogisticRegression, model_dict: dict, df: pd.DataFrame, label: str) -> None:
    means = pd.Series(model_dict["means"])
    stds = pd.Series(model_dict["stds"]).replace(0, 1.0)
    x = (df[FEATURE_NAMES] - means) / stds
    y = df["target_up"]
    proba = model.predict_proba(x)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(
        f"{label}: accuracy={accuracy_score(y, pred):.4f}, "
        f"auc={roc_auc_score(y, proba):.4f}, samples={len(df)}"
    )


def patch_trader_file(trader_path: Path, model_dict: dict) -> None:
    source = trader_path.read_text()
    start = source.index(BEGIN_MARKER)
    end = source.index(END_MARKER)
    replacement = (
        f"{BEGIN_MARKER}\n"
        f"MODEL = {json.dumps(model_dict, indent=4)}\n"
        f"{END_MARKER}"
    )
    updated = source[:start] + replacement + source[end + len(END_MARKER):]
    trader_path.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TOMATOES logistic regression model and patch trader_logreg.py.")
    parser.add_argument("--horizon", type=int, default=3, help="Prediction horizon in ticks.")
    parser.add_argument("--fit-all", action="store_true", help="Refit on train+validate before exporting model.")
    args = parser.parse_args()

    train_raw = load_prices(DEFAULT_TRAIN)
    validate_raw = load_prices(DEFAULT_VALIDATE)
    train_df = add_features(train_raw, args.horizon)
    validate_df = add_features(validate_raw, args.horizon)

    means = train_df[FEATURE_NAMES].mean()
    stds = train_df[FEATURE_NAMES].std().replace(0, 1.0).fillna(1.0)
    x_train = (train_df[FEATURE_NAMES] - means) / stds
    y_train = train_df["target_up"]

    model = LogisticRegression(max_iter=1000, C=0.5)
    model.fit(x_train, y_train)

    model_dict = {
        "feature_names": FEATURE_NAMES,
        "means": {name: float(means[name]) for name in FEATURE_NAMES},
        "stds": {name: float(stds[name]) for name in FEATURE_NAMES},
        "coefficients": {name: float(model.coef_[0][i]) for i, name in enumerate(FEATURE_NAMES)},
        "intercept": float(model.intercept_[0]),
        "horizon": int(args.horizon),
    }

    evaluate(model, model_dict, train_df, "train")
    evaluate(model, model_dict, validate_df, "validate")

    if args.fit_all:
        combined_raw = pd.concat([train_raw, validate_raw], ignore_index=True)
        combined_df = add_features(combined_raw, args.horizon)
        means = combined_df[FEATURE_NAMES].mean()
        stds = combined_df[FEATURE_NAMES].std().replace(0, 1.0).fillna(1.0)
        x_all = (combined_df[FEATURE_NAMES] - means) / stds
        y_all = combined_df["target_up"]

        model = LogisticRegression(max_iter=1000, C=0.5)
        model.fit(x_all, y_all)
        model_dict = {
            "feature_names": FEATURE_NAMES,
            "means": {name: float(means[name]) for name in FEATURE_NAMES},
            "stds": {name: float(stds[name]) for name in FEATURE_NAMES},
            "coefficients": {name: float(model.coef_[0][i]) for i, name in enumerate(FEATURE_NAMES)},
            "intercept": float(model.intercept_[0]),
            "horizon": int(args.horizon),
        }
        print(f"refit_all: samples={len(combined_df)}")

    MODEL_JSON.write_text(json.dumps(model_dict, indent=4))
    patch_trader_file(TRADER_FILE, model_dict)
    print(f"Wrote model to {MODEL_JSON}")
    print(f"Patched {TRADER_FILE}")


if __name__ == "__main__":
    main()

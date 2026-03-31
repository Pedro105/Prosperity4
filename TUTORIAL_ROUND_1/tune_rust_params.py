import argparse
import ast
import csv
import itertools
import json
import subprocess
import sys
import sysconfig
import time
from pathlib import Path


WORKSPACE = Path("/Users/pedropinto/Downloads/TUTORIAL_ROUND_1")
RUST_BACKTESTER = Path("/Users/pedropinto/Downloads/prosperity_rust_backtester-main")
RUST_RUNS_DIR = WORKSPACE / "rust_runs"
TRADER_FILE = WORKSPACE / "trader123.py"
BEGIN_MARKER = "# BEGIN PARAMS"
END_MARKER = "# END PARAMS"


def load_param_block(trader_path: Path) -> tuple[dict, str, str]:
    source = trader_path.read_text()
    start = source.index(BEGIN_MARKER)
    end = source.index(END_MARKER)
    prefix = source[:start]
    block = source[start:end]
    suffix = source[end:]

    _, dict_source = block.split("PARAMS =", 1)
    params = ast.literal_eval(dict_source.strip())
    return params, prefix, suffix


def render_param_block(params: dict) -> str:
    return (
        f"{BEGIN_MARKER}\n"
        f"PARAMS = {json.dumps(params, indent=4)}\n"
        f"{END_MARKER}\n"
    )


def set_nested(params: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    current = params
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def run_command(command: str, cwd: Path) -> None:
    completed = subprocess.run(["bash", "-lc", command], cwd=cwd, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {command}")


def run_variant(trader_path: Path, run_id: str, dataset: str, day: str | None) -> Path:
    python_executable = Path(sys.executable).resolve()
    python_libdir = Path(sysconfig.get_config_var("LIBDIR")).resolve()
    cargo_target_dir = RUST_RUNS_DIR / "target-py" / f"{sys.version_info.major}.{sys.version_info.minor}"

    day_arg = f" --day={day}" if day is not None else ""
    command = (
        f'source "$HOME/.cargo/env" && '
        f'PYO3_PYTHON="{python_executable}" '
        f'DYLD_FALLBACK_LIBRARY_PATH="{python_libdir}:${{DYLD_FALLBACK_LIBRARY_PATH:-}}" '
        f'CARGO_TARGET_DIR="{cargo_target_dir}" '
        f'./scripts/cargo_local.sh run -- '
        f'--trader "{trader_path}" '
        f'--dataset "{dataset}" '
        f'--artifact-mode none '
        f'--flat '
        f'--run-id "{run_id}" '
        f'--output-root "{RUST_RUNS_DIR}"'
        f'{day_arg}'
    )
    run_command(command, RUST_BACKTESTER)
    return RUST_RUNS_DIR / run_id


def parse_metrics(run_dir: Path) -> dict:
    metrics = {}
    metric_files = sorted(run_dir.glob("*-metrics.json"))
    if not metric_files and (run_dir / "metrics.json").exists():
        metric_files = [run_dir / "metrics.json"]

    for metrics_file in metric_files:
        data = json.loads(metrics_file.read_text())
        label = (
            data.get("dataset_id")
            or metrics_file.stem.replace("-metrics", "")
            or metrics_file.stem
        )
        metrics[label] = data["final_pnl_total"]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep trader123.py parameters with the Rust backtester.")
    parser.add_argument("--dataset", default="tutorial", help="Rust backtester dataset alias/path.")
    parser.add_argument("--day", default=None, help="Optional day filter, e.g. -1.")
    parser.add_argument("--top", type=int, default=10, help="Number of top results to print.")
    parser.add_argument(
        "--profile",
        choices=["quick", "full"],
        default="quick",
        help="Use a smaller or larger parameter grid.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optionally stop after N candidates for a smoke test.",
    )
    args = parser.parse_args()

    params, prefix, suffix = load_param_block(TRADER_FILE)

    if args.profile == "quick":
        grid = {
            "TOMATOES.ema_alpha": [0.20, 0.25],
            "TOMATOES.micro_weight": [0.60, 0.70],
            "TOMATOES.entry_edge_mult": [1.25, 1.35],
            "TOMATOES.quote_edge_mult": [0.55, 0.65],
            "TOMATOES.reservation_inventory_skew": [0.12, 0.15],
        }
    else:
        grid = {
            "TOMATOES.ema_alpha": [0.20, 0.25, 0.30],
            "TOMATOES.micro_weight": [0.60, 0.70, 0.80],
            "TOMATOES.entry_edge_mult": [1.25, 1.35, 1.45],
            "TOMATOES.quote_edge_mult": [0.55, 0.65, 0.75],
            "TOMATOES.reservation_inventory_skew": [0.12, 0.15, 0.18],
        }

    tuning_dir = WORKSPACE / "tuning"
    tuning_dir.mkdir(exist_ok=True)
    candidates_dir = tuning_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)

    results = []
    grid_items = list(grid.items())
    total_runs = 1
    for _, choices in grid_items:
        total_runs *= len(choices)

    for idx, values in enumerate(itertools.product(*(choices for _, choices in grid_items)), start=1):
        if args.max_runs is not None and idx > args.max_runs:
            break
        candidate_params = json.loads(json.dumps(params))
        label_parts = []
        for (key, _), value in zip(grid_items, values):
            set_nested(candidate_params, key, value)
            label_parts.append(f"{key.split('.')[-1]}={value}")

        # Keep tomato weights complementary during the sweep.
        candidate_params["TOMATOES"]["ema_weight"] = round(1.0 - candidate_params["TOMATOES"]["micro_weight"], 2)

        candidate_path = candidates_dir / f"candidate_{idx:03d}.py"
        candidate_path.write_text(prefix + render_param_block(candidate_params) + suffix)

        run_id = f"tune-{int(time.time() * 1000)}-{idx:03d}"
        run_dir = run_variant(candidate_path, run_id, args.dataset, args.day)
        metrics = parse_metrics(run_dir)
        score = sum(metrics.values())

        results.append(
            {
                "candidate": candidate_path.name,
                "score": round(score, 2),
                "params": " | ".join(label_parts),
                **{name: round(value, 2) for name, value in metrics.items()},
            }
        )
        display_total = min(total_runs, args.max_runs) if args.max_runs is not None else total_runs
        print(f"[{idx}/{display_total}] {candidate_path.name}: score={score:.2f}")

    results.sort(key=lambda row: row["score"], reverse=True)

    output_csv = tuning_dir / "tuning_results.csv"
    fieldnames = list(results[0].keys()) if results else ["candidate", "score", "params"]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nTop results:")
    for row in results[: args.top]:
        print(f"{row['candidate']}: score={row['score']:.2f} | {row['params']}")
    print(f"\nSaved full results to {output_csv}")


if __name__ == "__main__":
    main()

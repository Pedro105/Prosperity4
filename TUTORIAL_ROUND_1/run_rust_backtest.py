import argparse
import json
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
REPO_ROOT = WORKSPACE.parent
RUST_BACKTESTER = REPO_ROOT / "prosperity_rust_backtester-main"
RESULTS_DIR = WORKSPACE / "results"
RUST_RUNS_DIR = RESULTS_DIR / "rust_runs"


def run_command(command: str, cwd: Path) -> None:
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=cwd,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def plot_bundle(bundle_path: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    with bundle_path.open() as f:
        bundle = json.load(f)

    pnl_series = bundle.get("pnl_series", [])
    if not pnl_series:
        return

    timestamps = [row["timestamp"] for row in pnl_series]
    totals = [row["total"] for row in pnl_series]

    products = {}
    for row in pnl_series:
        for product, value in row.get("by_product", {}).items():
            products.setdefault(product, []).append(value)

    colors = {
        "EMERALDS": "#2ecc71",
        "TOMATOES": "#e74c3c",
        "TOTAL": "#3498db",
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    for product, values in products.items():
        ax.plot(
            timestamps,
            values,
            label=product,
            linewidth=1.2,
            alpha=0.9,
            color=colors.get(product),
        )

    ax.plot(
        timestamps,
        totals,
        label="TOTAL",
        linewidth=1.8,
        linestyle="--",
        alpha=0.9,
        color=colors["TOTAL"],
    )
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_title(bundle_path.stem.replace("-bundle", "").replace("-", " "))
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("PnL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open() as f:
        return json.load(f)


def write_run_summary(metrics_rows: list[dict], output_path: Path) -> None:
    lines = [
        "| Dataset | Day | Total PnL | EMERALDS | TOMATOES | Own Trades | Ticks |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in metrics_rows:
        by_product = row.get("final_pnl_by_product", {})
        lines.append(
            "| {dataset} | {day} | {total:.2f} | {emeralds:.2f} | {tomatoes:.2f} | {trades} | {ticks} |".format(
                dataset=row.get("dataset_id", ""),
                day=row.get("day", ""),
                total=row.get("final_pnl_total", 0.0),
                emeralds=by_product.get("EMERALDS", 0.0),
                tomatoes=by_product.get("TOMATOES", 0.0),
                trades=row.get("own_trade_count", 0),
                ticks=row.get("tick_count", 0),
            )
        )

    output_path.write_text("\n".join(lines) + "\n")


def write_run_summary_csv(metrics_rows: list[dict], output_path: Path) -> None:
    lines = ["dataset_id,day,total_pnl,emeralds_pnl,tomatoes_pnl,own_trade_count,tick_count"]
    for row in metrics_rows:
        by_product = row.get("final_pnl_by_product", {})
        lines.append(
            "{dataset},{day},{total:.2f},{emeralds:.2f},{tomatoes:.2f},{trades},{ticks}".format(
                dataset=row.get("dataset_id", ""),
                day=row.get("day", ""),
                total=row.get("final_pnl_total", 0.0),
                emeralds=by_product.get("EMERALDS", 0.0),
                tomatoes=by_product.get("TOMATOES", 0.0),
                trades=row.get("own_trade_count", 0),
                ticks=row.get("tick_count", 0),
            )
        )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Rust backtester and generate PnL PNGs.")
    parser.add_argument(
        "--trader",
        default=str(WORKSPACE / "trader123.py"),
        help="Path to trader Python file.",
    )
    parser.add_argument(
        "--dataset",
        default="tutorial",
        help="Dataset alias/path for the Rust backtester.",
    )
    args = parser.parse_args()

    trader_path = Path(args.trader).expanduser().resolve()
    trader_name = trader_path.stem
    run_id = f"cursor-{int(time.time() * 1000)}"
    output_root = RUST_RUNS_DIR
    run_results_dir = RESULTS_DIR / trader_name / run_id
    python_executable = Path(sys.executable).resolve()
    python_libdir = Path(sysconfig.get_config_var("LIBDIR")).resolve()
    cargo_target_dir = output_root / "target-py" / f"{sys.version_info.major}.{sys.version_info.minor}"
    run_results_dir.mkdir(parents=True, exist_ok=True)

    command = (
        f'source "$HOME/.cargo/env" && '
        f'PYO3_PYTHON="{python_executable}" '
        f'DYLD_FALLBACK_LIBRARY_PATH="{python_libdir}:${{DYLD_FALLBACK_LIBRARY_PATH:-}}" '
        f'CARGO_TARGET_DIR="{cargo_target_dir}" '
        f'./scripts/cargo_local.sh run -- '
        f'--trader "{trader_path}" '
        f'--dataset "{args.dataset}" '
        f'--artifact-mode diagnostic '
        f'--flat '
        f'--run-id "{run_id}" '
        f'--output-root "{output_root}"'
    )

    run_command(command, RUST_BACKTESTER)

    run_dir = output_root / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Expected run directory not found: {run_dir}")

    metrics_rows = []
    for metrics_path in sorted(run_dir.glob("*-metrics.json")):
        metrics_rows.append(load_metrics(metrics_path))

    run_summary_md = run_results_dir / f"{trader_name}-summary.md"
    run_summary_csv = run_results_dir / f"{trader_name}-summary.csv"
    write_run_summary(metrics_rows, run_summary_md)
    write_run_summary_csv(metrics_rows, run_summary_csv)
    print(f"Generated {run_summary_md}")
    print(f"Generated {run_summary_csv}")

    png_paths = []
    for bundle_path in sorted(run_dir.glob("*-bundle.json")):
        bundle_name = bundle_path.stem.replace("-bundle", "")
        png_path = run_results_dir / f"{trader_name}-{bundle_name}.png"
        plot_bundle(bundle_path, png_path)
        png_paths.append(png_path)
        print(f"Generated {png_path}")

    submission_png = next((p for p in png_paths if "submission" in p.name), None)
    latest_png = submission_png or (png_paths[-1] if png_paths else None)
    if latest_png is not None:
        shutil.copyfile(latest_png, WORKSPACE / "pnl_curve.png")
        print(f"Updated {latest_trader_png}")
        print(f"Updated {latest_summary_md}")
        print(f"Updated {latest_summary_csv}")
        print(f"Updated {WORKSPACE / 'pnl_curve.png'}")


if __name__ == "__main__":
    main()

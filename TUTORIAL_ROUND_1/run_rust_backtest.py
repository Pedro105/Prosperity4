import argparse
import json
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path


WORKSPACE = Path("/Users/pedropinto/Downloads/TUTORIAL_ROUND_1")
RUST_BACKTESTER = Path("/Users/pedropinto/Downloads/prosperity_rust_backtester-main")
RUST_RUNS_DIR = WORKSPACE / "rust_runs"


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
    run_id = f"cursor-{int(time.time() * 1000)}"
    output_root = RUST_RUNS_DIR
    python_executable = Path(sys.executable).resolve()
    python_libdir = Path(sysconfig.get_config_var("LIBDIR")).resolve()
    cargo_target_dir = output_root / "target-py" / f"{sys.version_info.major}.{sys.version_info.minor}"

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

    png_paths = []
    for bundle_path in sorted(run_dir.glob("*-bundle.json")):
        png_path = WORKSPACE / f"{bundle_path.stem.replace('-bundle', '')}.png"
        plot_bundle(bundle_path, png_path)
        png_paths.append(png_path)
        print(f"Generated {png_path}")

    submission_png = next((p for p in png_paths if "submission" in p.name), None)
    latest_png = submission_png or (png_paths[-1] if png_paths else None)
    if latest_png is not None:
        shutil.copyfile(latest_png, WORKSPACE / "pnl_curve.png")
        print(f"Updated {WORKSPACE / 'pnl_curve.png'}")


if __name__ == "__main__":
    main()

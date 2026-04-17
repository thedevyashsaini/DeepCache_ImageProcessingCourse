import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    return float(value)


def plot_runtime_per_prompt(
    per_prompt_rows: list[dict],
    out_dir: Path,
    default_interval: int,
    default_branch: int,
):
    base = [r for r in per_prompt_rows if r["mode"] == "baseline"]
    dc = [
        r
        for r in per_prompt_rows
        if r["mode"] == "deepcache_fixed"
        and int(r["cache_interval"]) == default_interval
        and int(r["cache_branch_id"]) == default_branch
    ]

    prompt_ids = sorted({int(r["prompt_index"]) for r in base})
    x = np.arange(len(prompt_ids))
    base_means = []
    dc_means = []
    for p in prompt_ids:
        b = [to_float(r["runtime_s"]) for r in base if int(r["prompt_index"]) == p]
        d = [to_float(r["runtime_s"]) for r in dc if int(r["prompt_index"]) == p]
        base_means.append(np.mean(b) if b else 0.0)
        dc_means.append(np.mean(d) if d else 0.0)

    width = 0.38
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, base_means, width=width, label="Baseline")
    plt.bar(
        x + width / 2,
        dc_means,
        width=width,
        label=f"DeepCache i={default_interval}, b={default_branch}",
    )
    plt.xticks(x, [f"P{p}" for p in prompt_ids])
    plt.ylabel("Runtime (s)")
    plt.title("Runtime per Prompt")
    plt.legend()
    plt.tight_layout()
    plt.savefig((out_dir / "runtime_per_prompt.png").as_posix(), dpi=200)
    plt.close()


def plot_speedup_per_prompt(
    per_prompt_rows: list[dict],
    out_dir: Path,
    default_interval: int,
    default_branch: int,
):
    dc = [
        r
        for r in per_prompt_rows
        if r["mode"] == "deepcache_fixed"
        and int(r["cache_interval"]) == default_interval
        and int(r["cache_branch_id"]) == default_branch
    ]
    prompt_ids = sorted({int(r["prompt_index"]) for r in dc})
    values = []
    for p in prompt_ids:
        arr = [
            to_float(r["speedup_vs_baseline"])
            for r in dc
            if int(r["prompt_index"]) == p
        ]
        values.append(np.mean(arr) if arr else 0.0)

    x = np.arange(len(prompt_ids))
    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(x, values)
    plt.xticks(x, [f"P{p}" for p in prompt_ids])
    plt.ylabel("Speedup (x)")
    plt.title(f"Speedup per Prompt (i={default_interval}, b={default_branch})")
    for bar, v in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.2f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig((out_dir / "speedup_per_prompt.png").as_posix(), dpi=200)
    plt.close()


def plot_interval_sweep(
    config_summary_rows: list[dict], out_dir: Path, default_branch: int
):
    rows = [
        r
        for r in config_summary_rows
        if r["mode"] == "deepcache_fixed"
        and int(r["cache_branch_id"]) == default_branch
    ]
    rows = sorted(rows, key=lambda x: int(x["cache_interval"]))
    intervals = [int(r["cache_interval"]) for r in rows]
    speedups = [to_float(r["avg_speedup_vs_baseline"]) for r in rows]

    plt.figure(figsize=(8.8, 4.8))
    plt.plot(intervals, speedups, marker="o")
    plt.xlabel("Cache Interval")
    plt.ylabel("Average Speedup (x)")
    plt.title(f"Interval Sweep (branch={default_branch})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig((out_dir / "interval_sweep.png").as_posix(), dpi=200)
    plt.close()


def plot_branch_sensitivity(
    config_summary_rows: list[dict], out_dir: Path, default_interval: int
):
    rows = [
        r
        for r in config_summary_rows
        if r["mode"] == "deepcache_fixed"
        and int(r["cache_interval"]) == default_interval
    ]
    rows = sorted(rows, key=lambda x: int(x["cache_branch_id"]))
    branches = [int(r["cache_branch_id"]) for r in rows]
    speedups = [to_float(r["avg_speedup_vs_baseline"]) for r in rows]
    l2s = [to_float(r["avg_l2_vs_baseline"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(9.2, 5.0))
    ax1.plot(branches, speedups, marker="o", label="Avg speedup")
    ax1.set_xlabel("Cache Branch ID")
    ax1.set_ylabel("Speedup (x)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(branches, l2s, marker="s", linestyle="--", label="Avg L2 vs baseline")
    ax2.set_ylabel("L2 (lower is better)")

    ax1.set_title(f"Layer Sensitivity (interval={default_interval})")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig((out_dir / "branch_sensitivity.png").as_posix(), dpi=200)
    plt.close(fig)


def plot_quality_speed_scatter(config_summary_rows: list[dict], out_dir: Path):
    rows = [r for r in config_summary_rows if r["mode"] == "deepcache_fixed"]
    x = [to_float(r["avg_speedup_vs_baseline"]) for r in rows]
    y = [to_float(r["avg_l2_vs_baseline"]) for r in rows]
    labels = [f"i{r['cache_interval']}-b{r['cache_branch_id']}" for r in rows]

    plt.figure(figsize=(8.8, 5.0))
    plt.scatter(x, y)
    for xi, yi, label in zip(x, y, labels):
        plt.text(xi, yi, label, fontsize=8)
    plt.xlabel("Average Speedup (x)")
    plt.ylabel("Average L2 vs Baseline")
    plt.title("Quality-Speed Trade-off (Current Fixed DeepCache)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig((out_dir / "quality_vs_speed_scatter.png").as_posix(), dpi=200)
    plt.close()


def plot_config_summary(
    config_summary_rows: list[dict],
    out_dir: Path,
    default_interval: int,
    default_branch: int,
):
    baseline = [r for r in config_summary_rows if r["mode"] == "baseline"]
    default_dc = [
        r
        for r in config_summary_rows
        if r["mode"] == "deepcache_fixed"
        and int(r["cache_interval"]) == default_interval
        and int(r["cache_branch_id"]) == default_branch
    ]
    if not baseline or not default_dc:
        return

    labels = ["Baseline", f"DeepCache i={default_interval}, b={default_branch}"]
    runtimes = [
        to_float(baseline[0]["avg_runtime_s"]),
        to_float(default_dc[0]["avg_runtime_s"]),
    ]
    speedups = [
        to_float(baseline[0]["avg_speedup_vs_baseline"]),
        to_float(default_dc[0]["avg_speedup_vs_baseline"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    axes[0].bar(labels, runtimes)
    axes[0].set_title("Average Runtime")
    axes[0].set_ylabel("Seconds")

    axes[1].bar(labels, speedups)
    axes[1].set_title("Average Speedup vs Baseline")
    axes[1].set_ylabel("x")

    fig.tight_layout()
    fig.savefig((out_dir / "avg_runtime_speedup.png").as_posix(), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot current DeepCache benchmark")
    parser.add_argument(
        "--input_tables", type=str, default="results/benchmarks/current/tables"
    )
    parser.add_argument(
        "--output_plots", type=str, default="results/benchmarks/current/plots"
    )
    parser.add_argument("--default_interval", type=int, default=3)
    parser.add_argument("--default_branch", type=int, default=0)
    args = parser.parse_args()

    input_tables = Path(args.input_tables)
    out_dir = Path(args.output_plots)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_prompt = read_csv(input_tables / "per_prompt_metrics.csv")
    summary = read_csv(input_tables / "config_summary.csv")

    plot_runtime_per_prompt(
        per_prompt, out_dir, args.default_interval, args.default_branch
    )
    plot_speedup_per_prompt(
        per_prompt, out_dir, args.default_interval, args.default_branch
    )
    plot_config_summary(summary, out_dir, args.default_interval, args.default_branch)
    plot_interval_sweep(summary, out_dir, args.default_branch)
    plot_branch_sensitivity(summary, out_dir, args.default_interval)
    plot_quality_speed_scatter(summary, out_dir)


if __name__ == "__main__":
    main()

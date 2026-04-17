import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def setup_hf_auth() -> None:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def ensure_dirs(base_dir: Path) -> dict:
    paths = {
        "root": base_dir,
        "raw": base_dir / "raw",
        "tables": base_dir / "tables",
        "plots": base_dir / "plots",
        "images": base_dir / "images",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def tensor_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).reshape(-1)
    return float(torch.sqrt(torch.mean(diff * diff)).item())


def tensor_psnr(a: torch.Tensor, b: torch.Tensor, max_value: float = 1.0) -> float:
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(max_value) - 10.0 * np.log10(mse))


def load_pipeline(model_id: str, token: str = None):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        token=token,
    )
    pipe = pipe.to("cuda:0")
    pipe.enable_attention_slicing()
    return pipe


def run_pipe(pipe, prompt: str, steps: int) -> tuple[torch.Tensor, float]:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.perf_counter()
    img = pipe(prompt, num_inference_steps=steps, output_type="pt").images[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return img.detach().cpu(), elapsed


def run_baseline(
    pipe, prompt: str, seed: int, steps: int
) -> tuple[torch.Tensor, float]:
    set_seed(seed)
    return run_pipe(pipe, prompt, steps)


def run_deepcache(
    pipe,
    prompt: str,
    seed: int,
    steps: int,
    cache_interval: int,
    cache_branch_id: int,
) -> tuple[torch.Tensor, float]:
    from DeepCache import DeepCacheSDHelper

    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(cache_interval=cache_interval, cache_branch_id=cache_branch_id)
    helper.enable()
    try:
        set_seed(seed)
        return run_pipe(pipe, prompt, steps)
    finally:
        helper.disable()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark current DeepCache setup")
    parser.add_argument("--config", type=str, default="benchmark_config_current.json")
    parser.add_argument("--output_root", type=str, default="results/benchmarks/current")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = ensure_dirs(Path(args.output_root))

    setup_hf_auth()
    token = os.environ.get("HF_TOKEN")
    model_id = config["model_id"]
    prompts = config["prompts"]
    seeds = config["seeds"]
    steps = int(config["steps"])
    repeats = int(config.get("repeats", 1))
    warmup_steps = int(config.get("warmup_steps", 5))
    cache_intervals = config["cache_intervals"]
    cache_branch_ids = config["cache_branch_ids"]

    logger.info("Loading pipeline: %s", model_id)
    pipe = load_pipeline(model_id=model_id, token=token)

    logger.info("Warmup run...")
    set_seed(0)
    _ = pipe("warmup", num_inference_steps=warmup_steps, output_type="pt")
    torch.cuda.empty_cache()

    run_meta = {
        "experiment": config.get("experiment_name", "DeepCache Current Benchmark"),
        "date": datetime.now().isoformat(),
        "git_sha": None,
        "system": {
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "pytorch": torch.__version__,
        },
        "config": config,
    }

    per_prompt_rows = []
    baseline_cache = {}

    logger.info("Running baseline once per prompt/seed/repeat...")
    for seed in seeds:
        for prompt_i, prompt in enumerate(prompts):
            for repeat_i in range(repeats):
                baseline_img, baseline_time = run_baseline(pipe, prompt, seed, steps)
                key = (prompt, seed, repeat_i)
                baseline_cache[key] = {
                    "img": baseline_img,
                    "time": baseline_time,
                    "path": paths["images"]
                    / f"baseline_p{prompt_i + 1}_s{seed}_r{repeat_i + 1}.png",
                }
                save_image([baseline_img], baseline_cache[key]["path"].as_posix())
                per_prompt_rows.append(
                    {
                        "mode": "baseline",
                        "prompt_index": prompt_i + 1,
                        "prompt": prompt,
                        "seed": seed,
                        "repeat": repeat_i + 1,
                        "steps": steps,
                        "cache_interval": 0,
                        "cache_branch_id": -1,
                        "runtime_s": round(baseline_time, 6),
                        "speedup_vs_baseline": 1.0,
                        "l2_vs_baseline": 0.0,
                        "psnr_vs_baseline": 99.0,
                        "image_path": baseline_cache[key]["path"].as_posix(),
                    }
                )

    logger.info("Running DeepCache sweep...")
    for cache_interval in cache_intervals:
        for cache_branch_id in cache_branch_ids:
            logger.info(
                "Config interval=%s branch=%s",
                cache_interval,
                cache_branch_id,
            )
            for seed in seeds:
                for prompt_i, prompt in enumerate(prompts):
                    for repeat_i in range(repeats):
                        key = (prompt, seed, repeat_i)
                        baseline_data = baseline_cache[key]

                        dc_img, dc_time = run_deepcache(
                            pipe,
                            prompt,
                            seed,
                            steps,
                            cache_interval=cache_interval,
                            cache_branch_id=cache_branch_id,
                        )
                        speedup = baseline_data["time"] / dc_time
                        l2 = tensor_l2(dc_img, baseline_data["img"])
                        psnr = tensor_psnr(dc_img, baseline_data["img"])

                        image_path = (
                            paths["images"]
                            / f"dc_i{cache_interval}_b{cache_branch_id}_p{prompt_i + 1}_s{seed}_r{repeat_i + 1}.png"
                        )
                        save_image([dc_img], image_path.as_posix())

                        per_prompt_rows.append(
                            {
                                "mode": "deepcache_fixed",
                                "prompt_index": prompt_i + 1,
                                "prompt": prompt,
                                "seed": seed,
                                "repeat": repeat_i + 1,
                                "steps": steps,
                                "cache_interval": cache_interval,
                                "cache_branch_id": cache_branch_id,
                                "runtime_s": round(dc_time, 6),
                                "speedup_vs_baseline": round(speedup, 6),
                                "l2_vs_baseline": round(l2, 8),
                                "psnr_vs_baseline": round(psnr, 6),
                                "image_path": image_path.as_posix(),
                            }
                        )

    # Summaries
    config_summary = []
    grouped = {}
    for row in per_prompt_rows:
        key = (row["mode"], row["cache_interval"], row["cache_branch_id"])
        grouped.setdefault(key, []).append(row)

    for (mode, interval, branch), rows in grouped.items():
        runtimes = np.array([r["runtime_s"] for r in rows], dtype=np.float64)
        speedups = np.array([r["speedup_vs_baseline"] for r in rows], dtype=np.float64)
        l2s = np.array([r["l2_vs_baseline"] for r in rows], dtype=np.float64)
        psnrs = np.array([r["psnr_vs_baseline"] for r in rows], dtype=np.float64)
        config_summary.append(
            {
                "mode": mode,
                "cache_interval": interval,
                "cache_branch_id": branch,
                "num_samples": len(rows),
                "avg_runtime_s": round(float(runtimes.mean()), 6),
                "std_runtime_s": round(float(runtimes.std(ddof=0)), 6),
                "avg_speedup_vs_baseline": round(float(speedups.mean()), 6),
                "std_speedup_vs_baseline": round(float(speedups.std(ddof=0)), 6),
                "avg_l2_vs_baseline": round(float(l2s.mean()), 8),
                "avg_psnr_vs_baseline": round(float(psnrs.mean()), 6),
            }
        )

    config_summary.sort(
        key=lambda x: (x["mode"], x["cache_interval"], x["cache_branch_id"])
    )

    per_prompt_fields = [
        "mode",
        "prompt_index",
        "prompt",
        "seed",
        "repeat",
        "steps",
        "cache_interval",
        "cache_branch_id",
        "runtime_s",
        "speedup_vs_baseline",
        "l2_vs_baseline",
        "psnr_vs_baseline",
        "image_path",
    ]
    summary_fields = [
        "mode",
        "cache_interval",
        "cache_branch_id",
        "num_samples",
        "avg_runtime_s",
        "std_runtime_s",
        "avg_speedup_vs_baseline",
        "std_speedup_vs_baseline",
        "avg_l2_vs_baseline",
        "avg_psnr_vs_baseline",
    ]

    write_csv(
        paths["tables"] / "per_prompt_metrics.csv", per_prompt_rows, per_prompt_fields
    )
    write_csv(paths["tables"] / "config_summary.csv", config_summary, summary_fields)

    raw_payload = {
        "meta": run_meta,
        "per_prompt_metrics": per_prompt_rows,
        "config_summary": config_summary,
    }
    (paths["raw"] / "benchmark_current_raw.json").write_text(
        json.dumps(raw_payload, indent=2), encoding="utf-8"
    )

    final_report = {
        "meta": run_meta,
        "highlights": {
            "best_speedup_config": max(
                [x for x in config_summary if x["mode"] == "deepcache_fixed"],
                key=lambda x: x["avg_speedup_vs_baseline"],
            ),
            "default_compare": config["default_compare"],
        },
        "tables": {
            "per_prompt_csv": (paths["tables"] / "per_prompt_metrics.csv").as_posix(),
            "config_summary_csv": (paths["tables"] / "config_summary.csv").as_posix(),
        },
        "raw_json": (paths["raw"] / "benchmark_current_raw.json").as_posix(),
    }

    report_path = paths["raw"] / "benchmark_current_report.json"
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    logger.info("Benchmark complete: %s", report_path.as_posix())


if __name__ == "__main__":
    main()

# Adaptive DeepCache: Dynamic Feature Reuse for Controllable Speed-Quality Tradeoffs

## Video Presentations (Compulsory)

- **Phase 1:** https://drive.google.com/file/d/1KpHNLJkWk6fV5afMk4OEjy7s-aBMlcs5/view
- **Phase 2:** https://drive.google.com/file/d/1LNovhzjFKhVbi4moWMVd40KsbETTAHlR/view

This repository contains our Image Processing course project based on:
- replication of **DeepCache: Accelerating Diffusion Models for Free** (CVPR 2024), and
- our extension: **Adaptive DeepCache**, which replaces fixed cache refresh timing with a data-driven dynamic policy.

## Team

- Devyash Saini (202351030)
- Vaibhav Sharma (202351154)
- CSE, IIIT Vadodara

## Project Overview

Diffusion inference repeatedly runs U-Net for many denoising steps, which is expensive.
DeepCache speeds this up by reusing intermediate features with fixed controls:
- `cache_interval` (refresh every k steps)
- `cache_branch_id` (where caching starts in U-Net)

Our contribution, **Adaptive DeepCache**, makes refresh decisions per step using latent dynamics:
- computes latent change,
- normalizes via EMA,
- applies phase-aware thresholds (early/mid/late),
- enforces guardrails (minimum refresh spacing and forced refresh interval).

This creates a controllable speed-quality frontier instead of one static operating point.

## Repository Structure

- `DeepCache/extension/deepcache.py` - Adaptive helper logic integrated into DeepCache helper
- `replicate_deepcache.py` - Baseline replication script for original DeepCache behavior
- `benchmark_current.py` - Fixed-policy benchmark sweep
- `benchmark_adaptive.py` - Adaptive-policy benchmark and layer-sensitivity runs
- `plot_current.py` - Plotting for fixed benchmark outputs
- `plot_adaptive_compare.py` - Plotting for adaptive vs fixed comparison
- `benchmark_config_current.json` - Config for fixed benchmark
- `benchmark_config_adaptive.json` - Config for adaptive benchmark
- `results/` - Raw outputs, tables, plots, and generated images
- `report_ieee/` - Final IEEE-style report source, presentation source, and figure assets

## Environment Setup

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -e .
pip install torchvision numpy pandas matplotlib seaborn
```

If model access requires authentication, create `.env` in repo root:

```bash
HF_TOKEN=your_huggingface_token
```

## How To Run

### 1) Replicate original DeepCache result

```bash
python replicate_deepcache.py --steps 50 --cache_interval 3 --cache_branch_id 0 --output_dir results
```

Primary output:
- `results/replication_report.json`
- side-by-side and per-method images in `results/`

### 2) Run fixed-policy benchmark

```bash
python benchmark_current.py --config benchmark_config_current.json --output_root results/benchmarks/current
```

Then generate plots:

```bash
python plot_current.py --input_csv results/benchmarks/current/tables/per_prompt_metrics.csv --output_dir results/benchmarks/current/plots
```

### 3) Run adaptive benchmark

```bash
python benchmark_adaptive.py --config benchmark_config_adaptive.json --output_root results/benchmarks/adaptive
```

Then generate comparison plots:

```bash
python plot_adaptive_compare.py --current_summary results/benchmarks/current/tables/config_summary.csv --adaptive_summary results/benchmarks/adaptive/tables/adaptive_config_summary.csv --adaptive_detail results/benchmarks/adaptive/tables/adaptive_per_prompt_metrics.csv --output_dir results/benchmarks/adaptive/plots
```

## Adaptive Policy Configuration

Adaptive policies are defined in `benchmark_config_adaptive.json` under `adaptive_policies` with:
- `threshold_early`, `threshold_mid`, `threshold_late`
- `early_ratio`, `mid_ratio`
- `force_refresh_every`
- `min_refresh_interval`
- `ema_alpha`
- `use_relative_delta`

Default policy presets in this repo:
- `adaptive_speed`
- `adaptive_balanced`
- `adaptive_quality`

## Main Outputs

- Replication report: `results/replication_report.json`
- Fixed benchmark summary: `results/benchmarks/current/tables/config_summary.csv`
- Adaptive benchmark summary: `results/benchmarks/adaptive/tables/adaptive_config_summary.csv`
- Adaptive per-prompt metrics: `results/benchmarks/adaptive/tables/adaptive_per_prompt_metrics.csv`
- Figures and artifacts for submission: `report_ieee/`

## Notes

- Exact timing depends on GPU, CUDA, driver, and library versions.
- We used RTX 3050 Laptop GPU with FP16 and attention slicing for our reported runs.

## Reference

If you use or build on the original method, please cite the DeepCache paper:

```bibtex
@inproceedings{ma2023deepcache,
  title={DeepCache: Accelerating Diffusion Models for Free},
  author={Ma, Xinyin and Fang, Gongfan and Wang, Xinchao},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

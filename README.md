<div align="center">

# Flow Q-Learning: Inverting the Behavioral Flow

### Latent Noise Analysis for Offline Reinforcement Learning via Flow Q-Learning

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-PDF-red.svg)](#citation)

</div>

---

## Overview

This repository implements an **ODE inversion pipeline** for Flow Q-Learning (FQL) — a data-driven offline RL algorithm that uses a flow-matching policy to model complex, multimodal action distributions.

FQL trains a behavioral-cloning (BC) flow that transports a standard Gaussian noise space onto the behavioral action distribution via a learned vector field. This repository builds the **inverse map**: given a trained flow, a state, and an action, we integrate the vector field backward in time to recover the latent noise that produced (or could have produced) that action, and we study the distributional properties of the recovered noise across:

- **81 controlled dataset-perturbation conditions** (3 removal strategies × 9 removal fractions × 3 environments)
- **A single-mode deletion stress test** that pushes action-space divergence far beyond the perturbation grid
- **An out-of-support probe ladder** that tests the inverse map on actions and state–action pairings the flow never saw during training

For the full empirical findings and their implications, see the accompanying paper (citation below).

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Reproducing the Main Results](#reproducing-the-main-results)
  - [1. Train the base agents](#1-train-the-base-agents)
  - [2. Dataset subset-inversion grid](#2-dataset-subset-inversion-grid)
  - [3. Single-mode removal stress test](#3-single-mode-removal-stress-test)
  - [4. Out-of-support probe ladder](#4-out-of-support-probe-ladder)
  - [5. PCA overlay teaser figure](#5-pca-overlay-teaser-figure)
  - [6. Regenerate all paper figures](#6-regenerate-all-paper-figures)
- [Repository Structure](#repository-structure)
- [Notes on Data](#notes-on-data)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Installation

```bash
conda env create -n fql_env -f environment.yml
conda activate fql_env
```

## Usage

The main implementation of FQL is in [`agents/fql.py`](agents/fql.py). All analysis scripts described below operate on a trained agent's checkpoint directory (containing `flags.json` and `params_*.pkl`).

---

## Reproducing the Main Results

### 1. Train the base agents

The paper reports results on three OGBench environments, chosen to span different task families, action dimensionalities, and regularization strengths.

**Primary environment (antmaze, d = 8):**
```bash
python main.py --env_name=antmaze-large-navigate-singletask-v0 --agent.q_agg=min --agent.alpha=10
```

**Additional environments used for the cross-environment comparison:**
```bash
python main.py --env_name=cube-single-play-singletask-v0 --agent.alpha=300
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --agent.discount=0.995 --agent.alpha=30
```

Each run produces a checkpoint directory under `exp/fql/Debug/<run_name>/`.

---

### 2. Dataset subset-inversion grid

Runs the three removal strategies (random, quality-guided, state-region) crossed with nine removal fractions (10%–90%), inverts the removed transitions, and computes the full action-space and noise-space divergence suite.

```bash
python run_subset_analysis.py \
    --exp_dir  exp/fql/Debug/<antmaze_run> \
    --save_dir analysis/subset_grid
```

Repeat with `--exp_dir` pointing at the cube-single and humanoidmaze checkpoints to reproduce all **81 grid conditions**. Each run writes one JSON per (strategy, fraction) condition; these are consolidated into `consolidated_data*.txt` files, one per environment, used by all downstream analysis and plotting scripts.

---

### 3. Single-mode removal stress test

Progressively deletes the `+1` action mode of one dimension in 5% increments, driving action-space divergence far beyond anything reachable by the grid strategies.

```bash
python mode_removal_sweep.py \
    --exp_dir     exp/fql/Debug/<antmaze_run> \
    --x0_path     exp/fql/Debug/<antmaze_run>/x0_all.npy \
    --target_dim  0 \
    --mode_sign   1 \
    --n_steps     20 \
    --max_samples 10000 \
    --save_dir    analysis/mode_sweep
```

Produces `mode_sweep_metrics.json` along with divergence and distribution-shift figures. `x0_all.npy` is the exact training-time noise, recovered by replaying the training random-number-generator chain; it is required here to establish ground-truth (z, s, a) triples.

---

### 4. Out-of-support probe ladder

Constructs seven probe sets — in-support control, four levels of Gaussian action corruption, shuffled state–action pairs, and uniform random actions — inverts each through the trained flow, and scores every action against the Gaussian prior.

```bash
python ood_probe_ladder.py \
    --exp_dir  exp/fql/Debug/<antmaze_run> \
    --n_probe  2000 \
    --save_dir analysis/ood_ladder
```

Produces `ood_ladder_metrics.json` (containing the KS statistic, symmetric KL, tail mass, and AUROC for every probe) plus the four-panel ladder figure. This is the source of the AUROC values reported in the paper's out-of-distribution detection results.

Optional rungs, if a second dataset or checkpoint is available:
```bash
python ood_probe_ladder.py \
    --exp_dir        exp/fql/Debug/<antmaze_run> \
    --cross_env_name antmaze-large-explore-singletask-v0 \
    --alt_exp_dir    exp/fql/Debug/<antmaze_seed1_run> \
    --n_probe 2000 --save_dir analysis/ood_ladder_full
```

---

### 5. PCA overlay teaser figure

Projects recovered noise from in-support and out-of-support actions into a single common PCA basis, at three removal fractions, for direct visual comparison.

```bash
python pca_insupport_vs_ood.py \
    --exp_dir   exp/fql/Debug/<antmaze_run> \
    --strategy  quality \
    --fracs     0.1,0.3,0.5 \
    --ood_probe shuffled \
    --n 2000 \
    --save_dir  figures/
```

Use `--ood_probe gauss --sigma 0.5` or `--ood_probe uniform` to reproduce the other probe variants. A separate script producing independent (non-overlaid) in-support and OOD PCA panels is available in `pca_noise_visualization.py`.

---

### 6. Regenerate all paper figures

Reproduces every main-text figure directly from the logged JSON metrics, in one command:

```bash
python make_paper_figures.py \
    --antmaze   consolidated_data.txt \
    --cube      consolidated_data_cube_single_play.txt \
    --humanoid  consolidated_data_humanoidmaze.txt \
    --ood_json  analysis/ood_ladder/ood_ladder_metrics.json \
    --mode_json analysis/mode_sweep/mode_sweep_metrics.json \
    --outdir    paper_figures
```

Outputs `fig2_concept`, `fig4_crossenv`, `fig5_modesweep`, `fig6_ood_ladder`, and `fig7_anisotropy` as vector PDFs with embedded fonts, ready for direct inclusion in the paper.

---

## Repository Structure

```text
.
├── agents/
│   └── fql.py                      # FQL agent implementation
├── run_subset_analysis.py          # dataset-partitioning + inversion grid runner
├── subset_inversion_analysis.py    # partitioning strategies, divergence + correlation metrics
├── mode_removal_sweep.py           # single-mode deletion stress test
├── ood_probe_ladder.py             # out-of-support probe construction and scoring
├── pca_insupport_vs_ood.py         # common-basis PCA overlay figure (in-support vs OOD)
├── pca_noise_visualization.py      # separate in-support / OOD PCA figures
├── make_paper_figures.py           # reproduces all main-text figures from logged JSON
├── consolidated_data*.txt          # raw per-condition metric JSONs, one file per environment
├── analysis/                       # output directory for grid, mode-sweep, and ladder runs
├── paper_figures/                  # regenerated vector figures
└── environment.yml                 # conda environment specification
```

---

## Notes on Data

OGBench dataset files (e.g. `antmaze-large-navigate-v0.npz` and its `-val` counterpart) are downloaded automatically on first use of a given `--env_name`. No manual dataset preparation is required.

---

## Citation

If you use this code or build on these findings, please cite:

```bibtex
@inproceedings{your2027inverting,
  title     = {Inverting the Behavioral Flow: Latent Noise Analysis for
               Offline Reinforcement Learning via Flow Q-Learning},
  author    = {},
  booktitle = {},
  year      = {2027}
}
```

This work builds on Flow Q-Learning:

```bibtex
@inproceedings{park2025flow,
  title     = {Flow Q-Learning},
  author    = {Park, Seohong and Li, Qiyang and Levine, Sergey},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```

---

## Acknowledgments

This project is developed at Purdue University. It builds directly on the [Flow Q-Learning](https://github.com/seohongpark/fql) codebase and the [OGBench](https://github.com/seohongpark/ogbench) benchmark suite.

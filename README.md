# Inverting the Flow: Latent-Space Auditing for Offline Reinforcement Learning


## Overview

This repository contains code for Behavior-Flow Inversion of an FQL agent. A trained FQL flow is a bijective map from a latent Gaussian noise space to the behavioral action distribution. We integrate the flow backward in time to recover the latent noise that produced (or could have produced) a given action, and we study the distributional properties of the recovered noise across various dataset-perturbation conditions. We borrowed the FQL agent implementation from the original FQL repository.

## Installation

```bash
conda env create -n <env_name> -f environment.yml
conda activate <env_name>
```
## Reproducing the Main Results

### 1. Train FQL Agents on OGBench Environments

The paper reports results on three OGBench environments.

**Primary Environment (antmaze-large-navigate-singletask-v0):**
```bash
python main.py --env_name=antmaze-large-navigate-singletask-v0 --agent.q_agg=min --agent.alpha=10
```
Each run produces a checkpoint directory under `exp/fql/Debug/<run_name>/` (Let `<build_folder>` be this directory). This contains flags.json and params_*.pkl, the checkpoint every downstream script reads.

**Additional Environments:**
```bash
# cube-single-play-singletask-v0
python main.py --env_name=cube-single-play-singletask-v0 --agent.alpha=300
# humanoidmaze-medium-navigate-singletask-v0
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --agent.discount=0.995 --agent.alpha=30
```

For each of the following results, replace `<build_folder>` with the folder for the corresponding training run.

### 2. Out-of-support Experiments

The below script experiments for the following: in-support control, four levels of Gaussian action corruption, shuffled state–action pairs, and uniform random actions, inverts each through the trained flow, and scores every action against the Gaussian prior.

```bash
python ood_probe_ladder.py \
    --exp_dir  <build_folder> \
    --n_probe  2000 \
    --save_dir analysis/ood_ladder
```

**Output**: Produces `ood_ladder_metrics.json` containing the KS statistic, symmetric KL, tail mass, and AUROC for every experiment.

---

### 3. PCA Plot of Recovered Noise

Projects recovered noise from in-support and out-of-support actions into a single common PCA basis, at three removal fractions, for direct visual comparison.

```bash
python pca_insupport_vs_ood.py \
    --exp_dir   <build_folder> \
    --strategy  quality \
    --fracs     0.1,0.3,0.5 \
    --ood_probe shuffled \
    --n 2000 \
    --save_dir  figures/
```

**Output**: Figure 1 of the main paper.

---

## Notes on Data

OGBench dataset files (e.g. `antmaze-large-navigate-v0.npz` and its `-val` counterpart) are downloaded automatically on first use of a given `--env_name`. No manual dataset preparation is required.

---

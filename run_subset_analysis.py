"""
run_subset_analysis.py
======================
Drop-in runner for the Dataset Subset Inversion & Noise Correlation Analysis.
Place this file in the ROOT of the fql repository alongside train.py.

USAGE
-----
# Strategy 1 — random removal (baseline sanity check)
python run_subset_analysis.py \
    --exp_dir    exp/fql/Debug/seed0 \
    --env_name   cube-single-play-singletask-v0 \
    --strategy   random \
    --removal_frac 0.30 \
    --max_samples  10000 \
    --save_dir   analysis/random_30pct

# Strategy 2 — quality-based removal (low-Q transitions)
python run_subset_analysis.py \
    --exp_dir    exp/fql/Debug/seed0 \
    --env_name   cube-single-play-singletask-v0 \
    --strategy   quality \
    --removal_frac 0.30 \
    --max_samples  10000 \
    --save_dir   analysis/quality_30pct

# Strategy 3 — state-region removal (remove a spatial region)
python run_subset_analysis.py \
    --exp_dir    exp/fql/Debug/seed0 \
    --env_name   cube-single-play-singletask-v0 \
    --strategy   state_region \
    --region_dims 0,1 \
    --removal_frac 0.30 \
    --max_samples  10000 \
    --save_dir   analysis/region_30pct

# Sweep multiple fractions (for ablation study)
for frac in 0.10 0.25 0.50 0.70; do
    python run_subset_analysis.py \
        --exp_dir  exp/fql/Debug/seed0 \
        --strategy random \
        --removal_frac $frac \
        --save_dir analysis/sweep_${frac}
done
"""
import sys, os

# Always point to the fql repo root regardless of where the script is called from
FQL_ROOT = '/depot/gupta869/data/rittwik/fql'
sys.path.insert(0, FQL_ROOT)

import os
import sys
import json
import argparse
import numpy as np
import jax

# ── Import FQL modules ────────────────────────────────────────────
# Add fql root to path (already there if run from fql/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.flax_utils import restore_agent

# ── Import our analysis module ────────────────────────────────────
from subset_inversion_analysis import run_subset_inversion_analysis


# ================================================================
# Argument parser — every flag documented with type and default
# ================================================================

def get_args():
    p = argparse.ArgumentParser(
        description="Dataset Subset Inversion & Noise Correlation Analysis for FQL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Experiment location ───────────────────────────────────────
    p.add_argument(
        '--exp_dir', type=str, required=True,
        help=(
            "Path to your FQL experiment output directory. "
            "Must contain: flags.json, x0_all.npy, "
            "and at least one checkpoint (e.g. params_1000000/)."
        ),
    )
    p.add_argument(
        '--checkpoint_epoch', type=int, default=None,
        help=(
            "Which checkpoint to load (the integer after 'params_'). "
            "Defaults to None = loads the latest checkpoint found."
        ),
    )

    # ── Environment / dataset ─────────────────────────────────────
    p.add_argument(
        '--env_name', type=str, default=None,
        help=(
            "OGBench environment name, e.g. 'cube-single-play-singletask-v0'. "
            "If None, reads from flags.json in --exp_dir."
        ),
    )
    p.add_argument(
        '--frame_stack', type=int, default=None,
        help="Frame stack (must match what was used in training). Reads from flags.json if None.",
    )

    # ── Removal strategy ──────────────────────────────────────────
    p.add_argument(
        '--strategy', type=str, default='random',
        choices=['random', 'quality', 'state_region'],
        help=(
            "Which data removal strategy to use:\n"
            "  random       — uniform random removal (baseline)\n"
            "  quality      — remove lowest-Q(s,a) transitions\n"
            "  state_region — remove a spatial hypercube of state space"
        ),
    )
    p.add_argument(
        '--removal_frac', type=float, default=0.30,
        help=(
            "Fraction of (s,a) pairs to REMOVE from the dataset. "
            "Must be in (0, 1). See literature table in README for guidance. "
            "Recommended sweep: [0.10, 0.25, 0.50, 0.70]."
        ),
    )
    p.add_argument(
        '--region_dims', type=str, default='0,1',
        help=(
            "Comma-separated state dimension indices defining the removal "
            "region for strategy='state_region'. "
            "E.g. '0,1' removes samples in the top-30%% of dims 0 and 1."
        ),
    )

    # ── Sampling & compute ────────────────────────────────────────
    p.add_argument(
        '--max_samples', type=int, default=10000,
        help=(
            "Maximum number of transitions to load from the dataset. "
            "Caps memory usage. For d=4 action dim: 10k uses ~5MB. "
            "For high-dim observations: reduce to 5000."
        ),
    )
    p.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for reproducibility of partitioning and subsampling.",
    )

    # ── Output ────────────────────────────────────────────────────
    p.add_argument(
        '--save_dir', type=str, default='analysis/',
        help="Directory to save figures and JSON metrics.",
    )

    return p.parse_args()


# ================================================================
# Helper: find latest checkpoint in exp_dir
# ================================================================

def find_latest_checkpoint(exp_dir: str) -> int | None:
    """Scan exp_dir for params_XXXXXX.pkl files, return the largest epoch."""
    epochs = []
    for name in os.listdir(exp_dir):
        if name.startswith('params_') and name.endswith('.pkl'):
            try:
                epochs.append(int(name.replace('params_', '').replace('.pkl', '')))
            except ValueError:
                pass
    return max(epochs) if epochs else None


# ================================================================
# Helper: compute Q-values for the quality strategy
# ================================================================

def compute_q_values(agent, observations, actions, batch_size=512) -> np.ndarray:
    """
    Compute Q(s, a) for each (s, a) pair using the trained critic.
    Used by the 'quality' removal strategy.
    """
    import jax.numpy as jnp
    N = len(actions)
    q_vals = np.zeros(N, dtype=np.float32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        obs_b = jnp.array(observations[start:end])
        act_b = jnp.array(actions[start:end])
        qs    = agent.network.select('critic')(obs_b, actions=act_b)
        q_b   = np.array(qs.mean(axis=0))     # average over ensemble
        q_vals[start:end] = q_b
    return q_vals


# ================================================================
# Main
# ================================================================

def main():
    args = get_args()

    # ── Read flags from training run ─────────────────────────────
    flags_path = os.path.join(args.exp_dir, 'flags.json')
    if not os.path.exists(flags_path):
        raise FileNotFoundError(
            f"flags.json not found in {args.exp_dir}. "
            "Make sure --exp_dir points to a completed FQL training output."
        )
    with open(flags_path) as f:
        train_flags = json.load(f)

    env_name    = args.env_name    or train_flags['env_name']
    frame_stack = args.frame_stack or train_flags.get('frame_stack')
    agent_cfg   = train_flags.get('agent', {})    # ConfigDict snapshot

    print(f"\n{'='*60}")
    print(f"  FQL Subset Inversion Analysis")
    print(f"{'='*60}")
    print(f"  exp_dir:       {args.exp_dir}")
    print(f"  env_name:      {env_name}")
    print(f"  strategy:      {args.strategy}")
    print(f"  removal_frac:  {args.removal_frac:.0%}")
    print(f"  max_samples:   {args.max_samples}")
    print(f"{'='*60}\n")

    # ── Load environment & datasets ───────────────────────────────
    print("[1/5] Loading environment and datasets ...")
    _, _, train_dataset, val_dataset = make_env_and_datasets(
        env_name, frame_stack=frame_stack
    )

    # Wrap in ReplayBuffer (same as train.py)
    train_dataset = Dataset.create(**train_dataset)
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=train_dataset.size + 1
    )
    train_dataset.p_aug    = train_flags.get('p_aug')
    train_dataset.frame_stack = frame_stack

    # ── Restore agent ─────────────────────────────────────────────
    print("[2/5] Restoring agent checkpoint ...")
    epoch = args.checkpoint_epoch or find_latest_checkpoint(args.exp_dir)
    if epoch is None:
        raise FileNotFoundError(
            f"No checkpoints found in {args.exp_dir}. "
            "Run training with --save_interval before analysis."
        )
    print(f"  Loading checkpoint epoch {epoch} ...")

    # We need to re-create the agent first (same as train.py)
    import ml_collections
    from agents import agents as agent_classes

    example_batch = train_dataset.sample(1)

    # Load agent config from flags
    import importlib.util
    cfg_mod = importlib.util.spec_from_file_location(
        "fql_config", os.path.join(os.path.dirname(__file__), 'agents/fql.py')
    )
    fql_mod = importlib.util.module_from_spec(cfg_mod)
    cfg_mod.loader.exec_module(fql_mod)
    config = fql_mod.get_config()

    # Overwrite with saved flags where present
    for k, v in train_flags.get('agent', {}).items():
        if hasattr(config, k) and k not in ('ob_dims', 'action_dim'):
            try:
                config[k] = v
            except Exception:
                pass

    agent = agent_classes['fql'].create(
        seed              = train_flags.get('seed', 0),
        ex_observations   = example_batch['observations'],
        ex_actions        = example_batch['actions'],
        config            = config,
    )
    agent = restore_agent(agent, args.exp_dir, epoch)
    print(f"  Agent restored from {args.exp_dir}/params_{epoch}.pkl")

    # ── Verify x0_all.npy exists ─────────────────────────────────
    x0_path = os.path.join(args.exp_dir, 'x0_all.npy')
    if not os.path.exists(x0_path):
        raise FileNotFoundError(
            f"x0_all.npy not found in {args.exp_dir}.\n"
            "Make sure your training run included sample_training_x0 "
            "(i.e. you are using the modified fql.py with that method)."
        )
    print(f"  x0_all.npy found at {x0_path}")

    # ── Prepare Q-values for quality strategy ─────────────────────
    q_values = None
    if args.strategy == 'quality':
        print("[3/5] Computing Q-values for quality-based removal ...")
        idx_all = np.random.RandomState(args.seed).choice(
            train_dataset.size, args.max_samples, replace=False
        )
        batch_all = train_dataset.get_subset(idx_all)
        obs_all   = np.array(batch_all['observations'], dtype=np.float32)
        act_all   = np.array(batch_all['actions'],      dtype=np.float32)
        q_values  = compute_q_values(agent, obs_all, act_all)
        print(f"  Q-value range: [{q_values.min():.2f}, {q_values.max():.2f}]")
    else:
        print("[3/5] Q-values not needed for this strategy — skipping.")

    # ── Parse region dims ─────────────────────────────────────────
    region_dims = None
    if args.strategy == 'state_region':
        region_dims = [int(x) for x in args.region_dims.split(',')]
        print(f"  State-region removal on dims: {region_dims}")

    # ── Run analysis ──────────────────────────────────────────────
    print(f"\n[4/5] Running subset inversion analysis (strategy={args.strategy}) ...")
    os.makedirs(args.save_dir, exist_ok=True)

    results = run_subset_inversion_analysis(
        agent        = agent,
        dataset      = train_dataset,
        x0_path      = x0_path,
        strategy     = args.strategy,
        removal_frac = args.removal_frac,
        q_values     = q_values,
        region_dims  = region_dims,
        max_samples  = args.max_samples,
        save_dir     = args.save_dir,
        seed         = args.seed,
    )

    # ── Save JSON metrics ─────────────────────────────────────────
    print(f"\n[5/5] Saving metrics ...")
    def to_serializable(obj):
        """Recursively convert numpy types to Python natives for json.dump."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(i) for i in obj]
        return obj

    metrics_out = to_serializable({
        'strategy':     results['strategy'],
        'n_full':       results['n_full'],
        'n_subset':     results['n_subset'],
        'removal_frac': args.removal_frac,
        'divergence':   results['divergence'],
        'correlation':  results['correlation'],
    })

    json_path = os.path.join(
        args.save_dir,
        f'metrics_{args.strategy}_{args.removal_frac:.0%}.json'
    )
    with open(json_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)

    print(f"  Metrics saved: {json_path}")
    print(f"  Figure saved:  {args.save_dir}/subset_inversion_{args.strategy}.png")
    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

    
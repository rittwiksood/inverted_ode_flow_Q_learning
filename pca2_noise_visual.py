"""
pca_insupport_vs_ood.py
=======================
ONE figure: in-support and OOD recovered noise overlaid on the SAME
PCA axes, for ONE removal technique, at THREE removal percentages.

For each removal fraction rho (default 10/30/50%):
  * take the removed subset of the chosen strategy (e.g. quality),
  * IN-SUPPORT cloud : invert the subset's true (s, a) pairs,
  * OOD cloud        : invert the SAME states with an OOD probe
                       applied to the SAME actions
                       (shuffled | gauss | uniform),
  * plot both clouds in one panel, in a common PCA basis fit on a
    full-data control inversion, with prior 1/2/3-sigma rings.

A fourth panel overlays the ||z_hat||^2 ECDFs of all clouds against
the chi^2(d) reference -- the per-action OOD score, seen directly.

USAGE
  python pca_insupport_vs_ood.py \
      --exp_dir  exp/fql/Debug/<run> \
      --strategy quality \
      --fracs    0.1,0.3,0.5 \
      --ood_probe shuffled \
      --n 2000 --save_dir analysis/pca_overlay
  # gauss probe:  --ood_probe gauss --sigma 0.5
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------- shared helpers (same as pca_noise_visualization) ----

def invert_batched(agent, obs, acts, batch_size=256):
    import jax.numpy as jnp
    n, d = acts.shape
    out = np.zeros((n, d), dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        z = np.array(agent.compute_reverse_flow_actions(
            jnp.array(obs[s:e]), jnp.array(acts[s:e])))
        out[s:e] = z.astype(np.float32)
    return out


def _load_env_dataset(env_name, frame_stack):
    from envs.env_utils import make_env_and_datasets
    from utils.datasets import Dataset, ReplayBuffer
    _, _, train_dataset, _ = make_env_and_datasets(
        env_name, frame_stack=frame_stack)
    train_dataset = Dataset.create(**train_dataset)
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=train_dataset.size + 1)
    return train_dataset


def _load_agent(exp_dir, dataset):
    import importlib.util
    from utils.flax_utils import restore_agent
    from agents import agents as agent_classes
    flags = json.load(open(os.path.join(exp_dir, 'flags.json')))
    spec = importlib.util.spec_from_file_location(
        "fql_cfg", os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'agents/fql.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    config = mod.get_config()
    ex = dataset.sample(1)
    agent = agent_classes['fql'].create(
        flags.get('seed', 0), ex['observations'], ex['actions'], config)
    epochs = [int(n.replace('params_', '').replace('.pkl', ''))
              for n in os.listdir(exp_dir)
              if n.startswith('params_') and n.endswith('.pkl')]
    agent = restore_agent(agent, exp_dir, max(epochs))
    print(f"Agent restored (epoch {max(epochs)})")
    return agent, flags


def compute_q_values(agent, obs, acts, batch_size=512):
    import jax.numpy as jnp
    N = len(acts)
    q = np.zeros(N, dtype=np.float32)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        qs = agent.network.select('critic')(
            jnp.array(obs[s:e]), actions=jnp.array(acts[s:e]))
        q[s:e] = np.array(qs.mean(axis=0))
    return q


def make_ood_actions(acts, probe, sigma, rng, d):
    """Apply the OOD probe to a copy of the given in-support actions."""
    if probe == 'shuffled':
        return acts[rng.permutation(len(acts))]
    if probe == 'gauss':
        return np.clip(acts + sigma * rng.randn(*acts.shape)
                       .astype(np.float32), -1, 1)
    if probe == 'uniform':
        return rng.uniform(-1, 1, size=(len(acts), d)).astype(np.float32)
    raise ValueError(probe)


# ---------------- figure ----------------------------------------------

def overlay_figure(pairs, z_control, d, probe_label, save_path, seed=0):
    """
    pairs : list of (panel_title, z_insupport, z_ood)
    z_control : (n, d) control inversion used to fit the common basis.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(z_control)
    evr = pca.explained_variance_ratio_

    n_panels = len(pairs) + 1
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(5.4 * n_panels, 5.2))
    fig.suptitle(
        f"In-support vs OOD ({probe_label}) recovered noise "
        f"-- common control PCA basis",
        fontsize=13, fontweight='bold')

    C_IN, C_OOD = '#2a78d6', '#d65f2a'

    for ax, (title, z_in, z_ood) in zip(axes[:-1], pairs):
        p_in = pca.transform(z_in)
        p_ood = pca.transform(z_ood)
        ax.scatter(p_ood[:, 0], p_ood[:, 1], s=5, alpha=0.28,
                   color=C_OOD, label='OOD probe', rasterized=True)
        ax.scatter(p_in[:, 0], p_in[:, 1], s=5, alpha=0.28,
                   color=C_IN, label='in-support', rasterized=True)
        for k in (1, 2, 3):
            ax.add_patch(plt.Circle((0, 0), k, fill=False, color='k',
                                    lw=1.0 if k < 3 else 1.4,
                                    ls='--' if k < 3 else '-'))
        out_in = 100 * np.mean(np.linalg.norm(p_in, axis=1) > 3)
        out_ood = 100 * np.mean(np.linalg.norm(p_ood, axis=1) > 3)
        ks_in, _ = stats.kstest(z_in.flatten(), 'norm')
        ks_ood, _ = stats.kstest(z_ood.flatten(), 'norm')
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_aspect('equal')
        ax.set_title(f"{title}\n"
                     f"in: KS={ks_in:.3f}, {out_in:.1f}% >3\u03c3   "
                     f"ood: KS={ks_ood:.3f}, {out_ood:.1f}% >3\u03c3",
                     fontsize=9)
        ax.set_xlabel(f'PC1 ({evr[0]*100:.0f}%)')
        ax.set_ylabel(f'PC2 ({evr[1]*100:.0f}%)')
        ax.legend(fontsize=8, loc='upper left')

    # ---- ECDF panel of ||z||^2 ----------------------------------
    ax = axes[-1]
    xs = np.linspace(0, chi2.ppf(0.9999, d) * 4, 600)
    ax.plot(xs, chi2.cdf(xs, d), 'k-', lw=2,
            label=f'$\\chi^2({d})$ reference')
    shades_in = ['#9dc3ea', '#5b9bd5', '#1f5fa8']
    shades_ood = ['#f0b799', '#e58a5b', '#c14e17']
    for i, (title, z_in, z_ood) in enumerate(pairs):
        for z, col, tag in ((z_in, shades_in[i % 3], 'in'),
                            (z_ood, shades_ood[i % 3], 'ood')):
            sq = np.sort(np.sum(z ** 2, axis=1))
            ax.plot(sq, np.arange(1, len(sq) + 1) / len(sq),
                    color=col, lw=1.6,
                    label=f'{title} ({tag})')
    ax.set_xscale('log')
    ax.set_xlabel('$\|\hat{z}\|^2$ (log)')
    ax.set_ylabel('ECDF')
    ax.set_title('Per-action OOD score, all clouds')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  -> Figure saved: {save_path}")


# ---------------- main --------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_dir', required=True)
    p.add_argument('--env_name', default=None)
    p.add_argument('--strategy', default='quality',
                   choices=['quality', 'random'])
    p.add_argument('--fracs', default='0.1,0.3,0.5')
    p.add_argument('--ood_probe', default='shuffled',
                   choices=['shuffled', 'gauss', 'uniform'])
    p.add_argument('--sigma', type=float, default=0.5,
                   help='noise scale for --ood_probe gauss')
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--max_samples', type=int, default=10000)
    p.add_argument('--save_dir', default='analysis/pca_overlay')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    flags = json.load(open(os.path.join(args.exp_dir, 'flags.json')))
    env_name = args.env_name or flags['env_name']

    dataset = _load_env_dataset(env_name, flags.get('frame_stack'))
    agent, _ = _load_agent(args.exp_dir, dataset)

    rng = np.random.RandomState(args.seed)
    n_pool = min(dataset.size, args.max_samples)
    idx = rng.choice(dataset.size, n_pool, replace=False)
    batch = dataset.get_subset(idx)
    obs_pool = np.array(batch['observations'], dtype=np.float32)
    acts_pool = np.array(batch['actions'], dtype=np.float32)
    d = acts_pool.shape[1]
    print(f"Env {env_name}: pool n={n_pool}, d={d}")

    q_values = None
    if args.strategy == 'quality':
        print("Computing Q-values ...")
        q_values = compute_q_values(agent, obs_pool, acts_pool)

    # common-basis control: random in-support subset of the full pool
    cidx = rng.choice(n_pool, args.n, replace=False)
    print("Inverting control (basis fit) ...")
    z_control = invert_batched(agent, obs_pool[cidx], acts_pool[cidx])

    probe_label = (f"gauss \u03c3={args.sigma:g}"
                   if args.ood_probe == 'gauss' else args.ood_probe)

    pairs = []
    for frac in [float(x) for x in args.fracs.split(',') if x]:
        n_remove = int(n_pool * frac)
        if args.strategy == 'random':
            ridx = rng.choice(n_pool, n_remove, replace=False)
        else:
            ridx = np.argsort(q_values)[:n_remove]
        sub = rng.choice(ridx, min(args.n, len(ridx)), replace=False)
        obs_r, acts_r = obs_pool[sub], acts_pool[sub]

        print(f"[rho={frac:.0%}] inverting in-support subset "
              f"(n={len(sub)}) ...")
        z_in = invert_batched(agent, obs_r, acts_r)

        acts_ood = make_ood_actions(acts_r, args.ood_probe,
                                    args.sigma, rng, d)
        print(f"[rho={frac:.0%}] inverting OOD probe ...")
        z_ood = invert_batched(agent, obs_r, acts_ood)

        pairs.append((f"{args.strategy} {int(frac*100)}%",
                      z_in, z_ood))

    out = os.path.join(
        args.save_dir,
        f"pca_overlay_{args.strategy}_{args.ood_probe}.png")
    overlay_figure(pairs, z_control, d, probe_label, out,
                   seed=args.seed)
    print("Done.")


if __name__ == '__main__':
    main()
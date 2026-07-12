"""
pca_noise_visualization.py
==========================
PCA visualization of inversion-recovered noise, in a COMMON basis.

Two figures for the paper:
  MODE insupport : recovered noise from removed subsets at
                   rho in {0.1, 0.3, 0.5} (removal strategy selectable).
                   Expected: panels indistinguishable from the prior
                   -- the distributional normalizer, seen directly.
  MODE ood       : recovered noise from the OOD probe ladder
                   (control, gauss_0.3, gauss_1, shuffled_pairs,
                   uniform_actions).
                   Expected: visible distortion beyond the support.

DESIGN
  * PCA basis is fit ONCE on the control recovered noise; every
    condition and a fresh N(0, I_d) reference are projected into
    that same basis (panels are directly comparable).
  * Prior 1/2/3-sigma circles overlaid on every panel.
  * Eigenvalue spectrum panel: N(0, I) has all eigenvalues ~= 1;
    deviations summarize non-Gaussianity in one curve.
  * Equal n per condition; same seed => estimator effects cancel.

USAGE
  python pca_noise_visualization.py \
      --exp_dir  exp/fql/Debug/<run> \
      --mode     both \
      --strategy quality \
      --fracs    0.1,0.3,0.5 \
      --n 2000 --save_dir analysis/pca_noise
  # add --three_d for an extra 3-D scatter figure per mode
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ================================================================
# 1.  Shared helpers
# ================================================================

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


# ================================================================
# 2.  Condition builders (return dict name -> (obs, acts))
# ================================================================

def build_insupport_conditions(obs_pool, acts_pool, q_values,
                               strategy, fracs, n, rng):
    """
    For each removal fraction rho, reproduce the removed subset of the
    grid experiment and return n of its (s, a) pairs for inversion.
    """
    N = len(acts_pool)
    conds = {}
    for rho in fracs:
        n_remove = int(N * rho)
        if strategy == 'random':
            ridx = rng.choice(N, n_remove, replace=False)
        elif strategy == 'quality':
            ridx = np.argsort(q_values)[:n_remove]      # lowest-Q first
        else:
            raise ValueError(f"strategy {strategy} not supported here")
        sub = rng.choice(ridx, min(n, len(ridx)), replace=False)
        conds[f'{strategy} {int(rho*100)}%'] = (obs_pool[sub],
                                                acts_pool[sub])
    return conds


def build_ood_conditions(obs_pool, acts_pool, n, rng, d):
    """Control + selected OOD probe rungs (equal n, same states)."""
    idx = rng.choice(len(acts_pool), n, replace=False)
    obs_c, acts_c = obs_pool[idx], acts_pool[idx]
    conds = {'control (in-support)': (obs_c, acts_c)}

    for s in (0.3, 1.0):
        pert = np.clip(acts_c + s * rng.randn(*acts_c.shape)
                       .astype(np.float32), -1, 1)
        conds[f'gauss \u03c3={s:g}'] = (obs_c, pert)

    conds['shuffled pairs'] = (obs_c, acts_c[rng.permutation(n)])
    conds['uniform actions'] = (
        obs_c, rng.uniform(-1, 1, size=(n, d)).astype(np.float32))
    return conds


# ================================================================
# 3.  PCA figure
# ================================================================

def pca_figure(z_by_cond, d, title, save_path, three_d=False, seed=0):
    """
    z_by_cond : ordered dict  name -> (n, d) recovered noise.
                FIRST entry is the control / reference condition;
                the PCA basis is fit on it.
    """
    from sklearn.decomposition import PCA

    names = list(z_by_cond.keys())
    z_control = z_by_cond[names[0]]
    n_ref = len(z_control)
    rng = np.random.RandomState(seed)
    z_prior = rng.randn(n_ref, d).astype(np.float32)   # fresh N(0, I)

    # Common basis: fit on control recovered noise (centered).
    pca = PCA(n_components=min(d, 3)).fit(z_control)
    evr = pca.explained_variance_ratio_

    # Eigen-spectrum in the SAME basis for every condition:
    # variance of each condition's projection onto control PCs.
    pca_full = PCA(n_components=d).fit(z_control)

    n_panels = len(names) + 1                          # + prior panel
    ncols = min(n_panels, 3)
    nrows = int(np.ceil(n_panels / ncols)) + 1         # + spectrum row

    fig = plt.figure(figsize=(5.2 * ncols, 4.6 * nrows))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # ---- scatter panels --------------------------------------------
    def scatter_panel(ax, z, name, color):
        p = pca.transform(z)
        ax.scatter(p[:, 0], p[:, 1], s=4, alpha=0.25, color=color,
                   rasterized=True)
        for k in (1, 2, 3):                            # prior sigma rings
            c = plt.Circle((0, 0), k, fill=False, color='k',
                           lw=1.0 if k < 3 else 1.4,
                           ls='--' if k < 3 else '-')
            ax.add_patch(c)
        lim = 6.0
        frac_out = float(np.mean(np.linalg.norm(p[:, :2], axis=1) > 3))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title(f"{name}\n{100*frac_out:.1f}% beyond 3\u03c3",
                     fontsize=10)
        ax.set_xlabel(f'PC1 ({evr[0]*100:.0f}% var)')
        ax.set_ylabel(f'PC2 ({evr[1]*100:.0f}% var)')

    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(names)))
    axes = []
    scatter_panel(fig.add_subplot(nrows, ncols, 1), z_prior,
                  'N(0, I) prior (reference)', '#888888')
    for i, name in enumerate(names):
        ax = fig.add_subplot(nrows, ncols, i + 2)
        scatter_panel(ax, z_by_cond[name], name, colors[i])
        axes.append(ax)

    # ---- eigenvalue spectrum panel ---------------------------------
    ax = fig.add_subplot(nrows, 1, nrows)
    comps = pca_full.components_                       # (d, d)
    xs = np.arange(1, d + 1)
    ax.axhline(1.0, color='k', lw=1.4, label='N(0, I): all eigenvalues = 1')
    spec_prior = np.var(z_prior @ comps.T, axis=0)
    ax.plot(xs, spec_prior, 'o--', color='#888888', ms=4,
            label='prior sample')
    for i, name in enumerate(names):
        spec = np.var(z_by_cond[name] @ comps.T, axis=0)
        ax.plot(xs, spec, 'o-', color=colors[i], ms=4, label=name)
    ax.set_yscale('log')
    ax.set_xlabel('principal component (control basis)')
    ax.set_ylabel('variance along component (log)')
    ax.set_title('Eigenvalue spectrum in the common control basis')
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  -> Figure saved: {save_path}")

    # ---- optional 3-D ----------------------------------------------
    if three_d and d >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(6 * min(len(names) + 1, 3),
                                  5.5 * int(np.ceil((len(names) + 1) / 3))))
        fig.suptitle(title + ' (3-D)', fontsize=13, fontweight='bold')
        all_z = [('prior', z_prior, '#888888')] + [
            (nm, z_by_cond[nm], colors[i]) for i, nm in enumerate(names)]
        for i, (nm, z, col) in enumerate(all_z):
            ax = fig.add_subplot(int(np.ceil(len(all_z) / 3)),
                                 min(len(all_z), 3), i + 1,
                                 projection='3d')
            p = pca.transform(z)
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=3, alpha=0.2,
                       color=col)
            ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_zlim(-6, 6)
            ax.set_title(nm, fontsize=9)
            ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
        out3 = save_path.replace('.png', '_3d.png')
        plt.tight_layout()
        plt.savefig(out3, dpi=150, bbox_inches='tight')
        plt.savefig(out3.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  -> Figure saved: {out3}")


# ================================================================
# 4.  Q-values (for quality strategy)
# ================================================================

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


# ================================================================
# 5.  Main
# ================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_dir', required=True)
    p.add_argument('--env_name', default=None)
    p.add_argument('--mode', default='both',
                   choices=['insupport', 'ood', 'both'])
    p.add_argument('--strategy', default='quality',
                   choices=['quality', 'random'])
    p.add_argument('--fracs', default='0.1,0.3,0.5')
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--max_samples', type=int, default=10000)
    p.add_argument('--three_d', action='store_true')
    p.add_argument('--save_dir', default='analysis/pca_noise')
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

    fracs = [float(x) for x in args.fracs.split(',') if x]

    # ---------------- in-support figure -----------------------------
    if args.mode in ('insupport', 'both'):
        q_values = None
        if args.strategy == 'quality':
            print("Computing Q-values for quality strategy ...")
            q_values = compute_q_values(agent, obs_pool, acts_pool)

        conds = build_insupport_conditions(
            obs_pool, acts_pool, q_values, args.strategy, fracs,
            args.n, rng)

        z_by_cond = {}
        # control first: uniform random in-support subset
        cidx = rng.choice(n_pool, args.n, replace=False)
        print("[insupport] inverting control ...")
        z_by_cond['control (full data)'] = invert_batched(
            agent, obs_pool[cidx], acts_pool[cidx])
        for name, (o, a) in conds.items():
            print(f"[insupport] inverting {name} (n={len(a)}) ...")
            z_by_cond[name] = invert_batched(agent, o, a)

        pca_figure(
            z_by_cond, d,
            title=("Recovered noise in the control PCA basis -- "
                   "in-support removal subsets"),
            save_path=os.path.join(args.save_dir, 'pca_insupport.png'),
            three_d=args.three_d, seed=args.seed)

    # ---------------- OOD figure ------------------------------------
    if args.mode in ('ood', 'both'):
        conds = build_ood_conditions(obs_pool, acts_pool, args.n, rng, d)
        z_by_cond = {}
        for name, (o, a) in conds.items():
            print(f"[ood] inverting {name} (n={len(a)}) ...")
            z_by_cond[name] = invert_batched(agent, o, a)

        pca_figure(
            z_by_cond, d,
            title=("Recovered noise in the control PCA basis -- "
                   "OOD probe ladder"),
            save_path=os.path.join(args.save_dir, 'pca_ood.png'),
            three_d=args.three_d, seed=args.seed)

    print("\nDone.")


if __name__ == '__main__':
    main()
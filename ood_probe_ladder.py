"""
ood_probe_ladder.py
===================
Out-of-support probe ladder for FQL ODE inversion.

Tests whether the flow's distributional normalization (Findings 1-3)
breaks OUTSIDE the training support -- the regime required by the
OOD-action-detection use case (Section VIII).

PROBE LADDER (each rung shares obs/action dims with training env):
  rung 0  control            uniform random subset of training (s, a)
  rung 1  gauss-perturbed    (s, a + eps), eps ~ N(0, sigma^2 I),
                             sigma sweep {0.1, 0.3, 0.5, 1.0}
  rung 2  shuffled pairs     (s_i, a_{pi(i)}): marginals identical,
                             state-action coupling destroyed (Case 3 probe)
  rung 3  cross-dataset      (s, a) from another dataset variant of the
                             same env family (optional: --cross_env_name)
  rung 4  alt-policy         actions from a differently trained agent
                             at training states (optional: --alt_exp_dir)
  rung 5  uniform actions    a ~ Uniform[-1, 1]^d at training states

METRICS
  Action side (probe vs training actions):
    unbiased MMD^2 (primary), exact per-dim W1 (secondary),
    boundary-mass fraction (saturation confound tracking)
  Noise side (z_hat vs the PRIOR N(0, I) -- foreign actions have no
  paired training noise):
    KS statistic, symmetric KL (KDE, per-dim mean),
    W2 per-dim vs a fresh Gaussian reference sample,
    per-dim |mean| and |std - 1|
  Per-sample OOD score:
    ||z_hat||^2 against chi^2(d): tail fractions beyond the 95th / 99th
    percentiles, and AUROC of each rung vs the control using ||z_hat||^2
  Saturation filter:
    all noise metrics recomputed excluding probe actions with
    max_j |a_j| > 0.99, separating OOD response from clipping artifact.

USAGE
-----
python ood_probe_ladder.py \
    --exp_dir  exp/fql/Debug/sd000_s_10954661.0.20260606_171910 \
    --n_probe  2000 \
    --save_dir analysis/ood_ladder

# with the optional rungs:
python ood_probe_ladder.py \
    --exp_dir        exp/.../cube_double_run \
    --cross_env_name cube-double-noisy-singletask-v0 \
    --alt_exp_dir    exp/.../cube_double_run_seed1 \
    --n_probe 2000 --save_dir analysis/ood_ladder
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
from scipy.spatial.distance import cdist
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ================================================================
# 1.  Metrics (self-contained; no imports from other analysis files)
# ================================================================

def compute_mmd(X, Y, sigma=None, n_subsample=2000):
    """Unbiased MMD^2 with RBF kernel, median-heuristic bandwidth."""
    rng = np.random.RandomState(0)
    if len(X) > n_subsample:
        X = X[rng.choice(len(X), n_subsample, replace=False)]
    if len(Y) > n_subsample:
        Y = Y[rng.choice(len(Y), n_subsample, replace=False)]
    if sigma is None:
        all_pts = np.vstack([X, Y])
        pw = cdist(all_pts, all_pts, 'sqeuclidean')
        sigma = max(float(np.sqrt(np.median(pw[pw > 0]) / 2)), 1e-6)
    g = 1.0 / (2 * sigma ** 2)
    Kxx = np.exp(-g * cdist(X, X, 'sqeuclidean'))
    Kyy = np.exp(-g * cdist(Y, Y, 'sqeuclidean'))
    Kxy = np.exp(-g * cdist(X, Y, 'sqeuclidean'))
    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    mmd2 = (Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
            - 2 * Kxy.mean())
    return float(max(mmd2, 0.0))


def w1_per_dim_mean(X, Y):
    """Exact 1-D W1 per dimension (scipy), averaged across dims."""
    d = X.shape[1]
    return float(np.mean([
        stats.wasserstein_distance(X[:, j], Y[:, j]) for j in range(d)
    ]))


def w2_per_dim(X, Y):
    """Exact 1-D W2 per dimension via sorted quantile arrays.
    Use ONLY with len(X) == len(Y) (see Finding 7)."""
    d = X.shape[1]
    n = min(len(X), len(Y))
    out = []
    for j in range(d):
        xs = np.sort(X[:, j])[:n]
        ys = np.sort(Y[:, j])[:n]
        out.append(float(np.sqrt(np.mean((xs - ys) ** 2))))
    return np.array(out)


def kl_kde_sym(X, Y, n_points=500):
    """Symmetric KL via per-dim KDE (Scott bandwidth), averaged."""
    d = X.shape[1]
    kls = []
    for j in range(d):
        xj = X[:, j].astype(np.float64)
        yj = Y[:, j].astype(np.float64)
        lo = min(xj.min(), yj.min()) - 0.5
        hi = max(xj.max(), yj.max()) + 0.5
        grid = np.linspace(lo, hi, n_points)
        try:
            p = stats.gaussian_kde(xj)(grid)
            q = stats.gaussian_kde(yj)(grid)
        except np.linalg.LinAlgError:
            kls.append(0.0)
            continue
        eps = 1e-12
        p /= p.sum() + eps
        q /= q.sum() + eps
        kl = 0.5 * (np.sum(p * np.log((p + eps) / (q + eps))) +
                    np.sum(q * np.log((q + eps) / (p + eps))))
        kls.append(float(kl))
    return float(np.mean(kls))


def noise_metrics(z_hat, z_ref, label=""):
    """
    Compare recovered noise against the PRIOR.
      z_hat: (n, d) recovered noise
      z_ref: (n, d) fresh N(0, I) reference sample (same n)
    Returns dict of prior-referenced metrics.
    """
    n, d = z_hat.shape
    flat = z_hat.flatten()
    ks_stat, ks_pval = stats.kstest(flat, 'norm')
    kl_sym = kl_kde_sym(z_hat, z_ref)
    w2_dims = w2_per_dim(z_hat, z_ref)
    dim_mean = np.abs(z_hat.mean(axis=0))
    dim_std = np.abs(z_hat.std(axis=0) - 1.0)

    # ||z||^2 against chi^2(d)
    sq_norms = np.sum(z_hat ** 2, axis=1)
    tail_p = 1.0 - chi2.cdf(sq_norms, df=d)
    frac_beyond_95 = float(np.mean(tail_p < 0.05))
    frac_beyond_99 = float(np.mean(tail_p < 0.01))

    return dict(
        n=n,
        ks_stat=float(ks_stat),
        ks_pval=float(ks_pval),
        kl_noise_sym=kl_sym,
        w2_noise_mean=float(np.mean(w2_dims)),
        w2_noise_per_dim=w2_dims.tolist(),
        dim_mean_max=float(dim_mean.max()),
        dim_std_dev_max=float(dim_std.max()),
        sq_norms=sq_norms,                    # kept for AUROC / plots
        frac_beyond_95=frac_beyond_95,
        frac_beyond_99=frac_beyond_99,
    )


def auroc_vs_control(scores_probe, scores_control):
    """AUROC for separating probe from control using ||z||^2."""
    try:
        from sklearn.metrics import roc_auc_score
        y = np.concatenate([np.zeros(len(scores_control)),
                            np.ones(len(scores_probe))])
        s = np.concatenate([scores_control, scores_probe])
        return float(roc_auc_score(y, s))
    except ImportError:
        # Mann-Whitney U equivalent
        from scipy.stats import mannwhitneyu
        u, _ = mannwhitneyu(scores_probe, scores_control,
                            alternative='greater')
        return float(u / (len(scores_probe) * len(scores_control)))


# ================================================================
# 2.  Probe constructors
# ================================================================

def build_probes(obs, acts, rng, sigmas, d,
                 cross_obs=None, cross_acts=None,
                 alt_actions=None, clip_probes=True):
    """
    Build the probe ladder. Every probe: dict(name, rung, obs, acts).
    obs, acts: the control pool (n_probe rows each, from training data).
    """
    n = len(acts)
    probes = []

    # rung 0 -- control (in-support)
    probes.append(dict(name='control', rung=0, obs=obs, acts=acts))

    # rung 1 -- Gaussian-perturbed dataset actions (sigma sweep)
    for s in sigmas:
        pert = acts + s * rng.randn(*acts.shape).astype(np.float32)
        if clip_probes:
            pert = np.clip(pert, -1.0, 1.0)
        probes.append(dict(name=f'gauss_{s:g}', rung=1, sigma=s,
                           obs=obs, acts=pert.astype(np.float32)))

    # rung 2 -- shuffled pairs (identical marginals, broken coupling)
    perm = rng.permutation(n)
    probes.append(dict(name='shuffled_pairs', rung=2,
                       obs=obs, acts=acts[perm]))

    # rung 3 -- cross-dataset (optional)
    if cross_obs is not None and cross_acts is not None:
        probes.append(dict(name='cross_dataset', rung=3,
                           obs=cross_obs, acts=cross_acts))

    # rung 4 -- alt-policy actions at training states (optional)
    if alt_actions is not None:
        probes.append(dict(name='alt_policy', rung=4,
                           obs=obs, acts=alt_actions))

    # rung 5 -- uniform random actions at training states
    uni = rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float32)
    probes.append(dict(name='uniform_actions', rung=5,
                       obs=obs, acts=uni))

    return probes


# ================================================================
# 3.  Batched inversion
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


# ================================================================
# 4.  Main experiment
# ================================================================

def run_ladder(agent, obs_pool, acts_pool, sigmas, n_probe, seed,
               save_dir, cross_obs=None, cross_acts=None,
               alt_agent=None, clip_probes=True,
               saturation_thresh=0.99):
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    d = acts_pool.shape[1]

    # control pool
    idx = rng.choice(len(acts_pool), n_probe, replace=False)
    obs_c, acts_c = obs_pool[idx], acts_pool[idx]

    # rung 4 actions generated AT THE CONTROL STATES for alignment
    alt_actions = None
    if alt_agent is not None:
        import jax
        alt_actions = np.array(alt_agent.sample_actions(
            observations=obs_c, seed=jax.random.PRNGKey(seed)),
            dtype=np.float32)

    # fixed Gaussian reference for prior-referenced W2 / KL
    z_ref = rng.randn(n_probe, d).astype(np.float32)

    probes = build_probes(obs_c, acts_c, rng, sigmas, d,
                          cross_obs=cross_obs, cross_acts=cross_acts,
                          alt_actions=alt_actions,
                          clip_probes=clip_probes)

    results = []
    control_sq_norms = None

    for p in probes:
        name = p['name']
        print(f"\n[{name}] (rung {p['rung']})  n={len(p['acts'])}")

        # --- action-side distinctiveness vs training pool ------------
        mmd = compute_mmd(acts_pool, p['acts'])
        w1 = w1_per_dim_mean(acts_pool, p['acts'])
        boundary_frac = float(np.mean(
            np.max(np.abs(p['acts']), axis=1) > saturation_thresh))
        print(f"  action MMD^2={mmd:.4f}  W1={w1:.4f}  "
              f"boundary_frac={boundary_frac:.3f}")

        # --- inversion ------------------------------------------------
        z_hat = invert_batched(agent, p['obs'], p['acts'])

        # --- noise-side metrics vs PRIOR -------------------------------
        nm = noise_metrics(z_hat, z_ref)
        print(f"  noise  KS={nm['ks_stat']:.4f}  KL={nm['kl_noise_sym']:.4f}  "
              f"W2={nm['w2_noise_mean']:.4f}  "
              f"tail95={nm['frac_beyond_95']:.3f}  "
              f"tail99={nm['frac_beyond_99']:.3f}")

        # --- saturation-filtered variant -------------------------------
        keep = np.max(np.abs(p['acts']), axis=1) <= saturation_thresh
        nm_filt = None
        if keep.sum() >= 200 and keep.sum() < len(keep):
            nm_filt = noise_metrics(z_hat[keep], z_ref[:keep.sum()])
            print(f"  [filtered n={int(keep.sum())}]  "
                  f"KS={nm_filt['ks_stat']:.4f}  "
                  f"KL={nm_filt['kl_noise_sym']:.4f}")

        # --- AUROC vs control ------------------------------------------
        if name == 'control':
            control_sq_norms = nm['sq_norms']
            auroc = 0.5
        else:
            auroc = auroc_vs_control(nm['sq_norms'], control_sq_norms)
        print(f"  AUROC vs control: {auroc:.4f}")

        results.append(dict(
            name=name, rung=p['rung'], sigma=p.get('sigma'),
            n=len(p['acts']),
            action_mmd2=mmd, action_w1=w1,
            boundary_frac=boundary_frac,
            noise=nm, noise_filtered=nm_filt,
            auroc=auroc,
        ))

    _plot_ladder(results, d, save_dir)
    _save_json(results, save_dir)
    return results


# ================================================================
# 5.  Figures
# ================================================================

def _plot_ladder(results, d, save_dir):
    names = [r['name'] for r in results]
    ks = [r['noise']['ks_stat'] for r in results]
    kl = [r['noise']['kl_noise_sym'] for r in results]
    mmd = [max(r['action_mmd2'], 1e-6) for r in results]
    auroc = [r['auroc'] for r in results]
    t95 = [r['noise']['frac_beyond_95'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("OOD Probe Ladder -- Two-Regime Test of the "
                 "Distributional Normalizer", fontsize=13,
                 fontweight='bold')

    # (A) noise KS and KL per rung
    ax = axes[0, 0]
    x = np.arange(len(names))
    ax.bar(x - 0.2, ks, 0.4, label='KS stat', color='#2a78d6', alpha=0.85)
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, kl, 0.4, label='noise KL', color='#d65f2a', alpha=0.85)
    ax.axhline(ks[0], color='#2a78d6', lw=1.2, ls='--',
               label='in-support KS floor')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('KS statistic', color='#2a78d6')
    ax2.set_ylabel('symmetric KL', color='#d65f2a')
    ax.set_title('(A) Noise divergence from prior, per probe')
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=8)

    # (B) dose-response: action MMD^2 (log x) vs noise KL
    ax = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for i, r in enumerate(results):
        ax.scatter(mmd[i], kl[i], s=90, color=colors[i], zorder=3,
                   label=r['name'])
    ax.axhline(kl[0], color='grey', lw=1.2, ls='--')
    ax.axhspan(kl[0] * 0.8, kl[0] * 1.2, alpha=0.12, color='grey',
               label='in-support band')
    ax.set_xscale('log')
    ax.set_xlabel('action MMD$^2$ vs training pool (log)')
    ax.set_ylabel('noise symmetric KL vs prior')
    ax.set_title('(B) Dose-response across the support boundary')
    ax.legend(fontsize=7, ncol=2)

    # (C) ||z||^2 distributions vs chi^2(d)
    ax = axes[1, 0]
    xs = np.linspace(0, max(np.percentile(r['noise']['sq_norms'], 99.5)
                            for r in results) * 1.05, 400)
    ax.plot(xs, chi2.pdf(xs, df=d), 'k-', lw=2,
            label=f'$\\chi^2({d})$ reference')
    for i, r in enumerate(results):
        ax.hist(r['noise']['sq_norms'], bins=60, density=True,
                alpha=0.35, color=colors[i], label=r['name'])
    ax.set_xlabel('$‖\\hat{z}‖^2$')
    ax.set_ylabel('density')
    ax.set_title('(C) Recovered-noise squared norms vs prior')
    ax.legend(fontsize=7, ncol=2)

    # (D) AUROC + tail mass per rung
    ax = axes[1, 1]
    ax.bar(x - 0.2, auroc, 0.4, color='#1baf7a', alpha=0.85,
           label='AUROC vs control')
    ax.bar(x + 0.2, t95, 0.4, color='#7f77dd', alpha=0.85,
           label='tail mass beyond 95th pct')
    ax.axhline(0.5, color='k', lw=1, ls='--')
    ax.axhline(0.05, color='#7f77dd', lw=1, ls=':',
               label='5% nominal tail')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title('(D) Per-action OOD detectability')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(save_dir, 'ood_probe_ladder.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"\n  -> Figure saved: {out}")


def _save_json(results, save_dir):
    def ser(o):
        if isinstance(o, np.ndarray):
            return None                       # drop bulky arrays
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, dict):
            return {k: ser(v) for k, v in o.items() if k != 'sq_norms'}
        if isinstance(o, list):
            return [ser(i) for i in o]
        return o
    out = os.path.join(save_dir, 'ood_ladder_metrics.json')
    with open(out, 'w') as f:
        json.dump(ser(results), f, indent=2)
    print(f"  -> Metrics saved: {out}")


# ================================================================
# 6.  CLI
# ================================================================

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
    epoch = max(epochs)
    agent = restore_agent(agent, exp_dir, epoch)
    print(f"Agent restored from {exp_dir} (epoch {epoch})")
    return agent, flags


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_dir', required=True)
    p.add_argument('--env_name', default=None)
    p.add_argument('--cross_env_name', default=None,
                   help='optional rung-3 dataset (must match obs/action dims)')
    p.add_argument('--alt_exp_dir', default=None,
                   help='optional rung-4 alternative agent checkpoint')
    p.add_argument('--n_probe', type=int, default=2000)
    p.add_argument('--sigmas', default='0.1,0.3,0.5,1.0')
    p.add_argument('--max_samples', type=int, default=10000)
    p.add_argument('--no_clip_probes', action='store_true',
                   help='do not clip perturbed actions to [-1,1]')
    p.add_argument('--save_dir', default='analysis/ood_ladder')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    flags = json.load(open(os.path.join(args.exp_dir, 'flags.json')))
    env_name = args.env_name or flags['env_name']
    frame_stack = flags.get('frame_stack')

    # training dataset + agent
    dataset = _load_env_dataset(env_name, frame_stack)
    agent, _ = _load_agent(args.exp_dir, dataset)

    rng = np.random.RandomState(args.seed)
    n_pool = min(dataset.size, args.max_samples)
    idx = rng.choice(dataset.size, n_pool, replace=False)
    batch = dataset.get_subset(idx)
    print(batch)
    obs_pool = np.array(batch['observations'], dtype=np.float32)
    acts_pool = np.array(batch['actions'], dtype=np.float32)
    d = acts_pool.shape[1]
    print(f"Env {env_name}: pool n={n_pool}, action dim d={d}")

    # optional rung 3
    cross_obs = cross_acts = None
    if args.cross_env_name:
        cds = _load_env_dataset(args.cross_env_name, frame_stack)
        cidx = rng.choice(cds.size, args.n_probe, replace=False)
        cb = cds.get_subset(cidx)
        cross_obs = np.array(cb['observations'], dtype=np.float32)
        cross_acts = np.array(cb['actions'], dtype=np.float32)
        assert cross_obs.shape[1] == obs_pool.shape[1] and \
               cross_acts.shape[1] == d, \
               "cross env must match obs/action dims"
        print(f"Rung 3 enabled: {args.cross_env_name}")

    # optional rung 4
    alt_agent = None
    if args.alt_exp_dir:
        alt_agent, _ = _load_agent(args.alt_exp_dir, dataset)
        print(f"Rung 4 enabled: {args.alt_exp_dir}")

    sigmas = [float(s) for s in args.sigmas.split(',') if s]

    run_ladder(
        agent, obs_pool, acts_pool,
        sigmas=sigmas, n_probe=args.n_probe, seed=args.seed,
        save_dir=args.save_dir,
        cross_obs=cross_obs, cross_acts=cross_acts,
        alt_agent=alt_agent,
        clip_probes=not args.no_clip_probes,
    )


if __name__ == '__main__':
    main()
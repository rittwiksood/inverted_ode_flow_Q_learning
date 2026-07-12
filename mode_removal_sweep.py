"""
mode_removal_sweep.py
=====================
Removes progressively more of one action mode (+1) to push action KL
from 0 up to ~log(2), then detects where KL(action) vs KL(noise)
becomes non-linear.

USAGE
-----
python mode_removal_sweep.py \
    --exp_dir   exp/fql/Debug/sd000_s_10954661.0.20260606_171910 \
    --x0_path   exp/fql/Debug/sd000_s_10954661.0.20260606_171910/x0_all.npy \
    --target_dim 0 \
    --mode_sign  1 \
    --n_steps    20 \
    --max_samples 10000 \
    --save_dir   analysis/mode_sweep
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
from scipy.signal import argrelmin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ================================================================
# 1.  Mode Identification and Removal
# ================================================================

def find_mode_boundary(actions: np.ndarray, target_dim: int) -> float:
    """
    Find the boundary between the two modes in target_dim via
    the minimum density point between the two peaks (KDE valley).
    Falls back to 0.0 if no clear valley is found.
    """
    vals = actions[:, target_dim].astype(np.float64)
    kde  = stats.gaussian_kde(vals)
    xs   = np.linspace(vals.min(), vals.max(), 1000)
    ys   = kde(xs)

    mins = argrelmin(ys, order=20)[0]
    if len(mins) == 0:
        print(f"  [warn] No clear valley in dim {target_dim}; using 0.0 as boundary")
        return 0.0

    centre   = (vals.min() + vals.max()) / 2.0
    best_min = mins[np.argmin(np.abs(xs[mins] - centre))]
    boundary = float(xs[best_min])
    print(f"  Mode boundary in dim {target_dim}: {boundary:.4f}")
    return boundary


def mode_removal_sweep(
    observations: np.ndarray,
    actions:      np.ndarray,
    z_original:   np.ndarray,
    target_dim:   int = 0,
    mode_sign:    int = 1,
    n_steps:      int = 20,
    seed:         int = 42,
) -> list[dict]:
    """
    Sweep mode removal from 0% to 100% in n_steps steps.
    Returns a list of step dicts containing all arrays needed
    for metrics and plotting.
    """
    rng = np.random.RandomState(seed)

    boundary  = find_mode_boundary(actions, target_dim)
    mode_mask = (actions[:, target_dim] > boundary) if mode_sign > 0 \
                else (actions[:, target_dim] < boundary)

    mode_idx  = np.where(mode_mask)[0]
    other_idx = np.where(~mode_mask)[0]
    n_mode    = len(mode_idx)

    print(f"  Samples in target mode:  {n_mode} / {len(actions)}")
    print(f"  Samples in other mode:   {len(other_idx)} / {len(actions)}")

    steps        = []
    remove_fracs = np.linspace(0.0, 1.0, n_steps + 1)

    for frac in remove_fracs:
        n_remove = int(n_mode * frac)
        if n_remove == 0:
            remove_idx = np.array([], dtype=int)
            keep_idx   = np.arange(len(actions))
        else:
            remove_idx = rng.choice(mode_idx, size=n_remove, replace=False)
            keep_mask2 = np.ones(len(actions), dtype=bool)
            keep_mask2[remove_idx] = False
            keep_idx = np.where(keep_mask2)[0]

        steps.append(dict(
            remove_frac    = float(frac),
            n_remove       = n_remove,
            n_mode         = n_mode,
            keep_idx       = keep_idx,
            remove_idx     = remove_idx,
            actions_full   = actions[keep_idx],
            actions_subset = actions[remove_idx] if n_remove > 0 else actions[:0],
            z_subset       = z_original[remove_idx] if n_remove > 0 else z_original[:0],
            obs_subset     = observations[remove_idx] if n_remove > 0 else observations[:0],
        ))

    return steps


# ================================================================
# 2.  Divergence Metrics
# ================================================================

def kl_kde(X: np.ndarray, Y: np.ndarray, n_points: int = 500) -> float:
    """
    Symmetric KL divergence via KDE, averaged across dimensions.
    """
    d   = X.shape[1]
    kls = []
    for j in range(d):
        xj   = X[:, j].astype(np.float64)
        yj   = Y[:, j].astype(np.float64)
        lo   = min(xj.min(), yj.min()) - 0.5
        hi   = max(xj.max(), yj.max()) + 0.5
        grid = np.linspace(lo, hi, n_points)
        try:
            p = stats.gaussian_kde(xj)(grid)
            q = stats.gaussian_kde(yj)(grid)
        except np.linalg.LinAlgError:
            kls.append(0.0)
            continue
        eps = 1e-12
        p  /= p.sum() + eps
        q  /= q.sum() + eps
        kl  = 0.5 * (np.sum(p * np.log((p + eps) / (q + eps))) +
                     np.sum(q * np.log((q + eps) / (p + eps))))
        kls.append(float(kl))
    return float(np.mean(kls))


def invert_subset_batched(
    agent,
    obs:        np.ndarray,
    acts:       np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Run compute_reverse_flow_actions in batches."""
    import jax.numpy as jnp
    N   = len(acts)
    d   = acts.shape[1]
    out = np.zeros((N, d), dtype=np.float32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        z_b = np.array(
            agent.compute_reverse_flow_actions(
                jnp.array(obs[start:end]),
                jnp.array(acts[start:end]),
            )
        )
        out[start:end] = z_b.astype(np.float32)
    return out


# ================================================================
# 3.  Main Sweep Runner
# ================================================================

def run_mode_sweep(
    agent,
    dataset,
    x0_path:     str,
    target_dim:  int = 0,
    mode_sign:   int = 1,
    n_steps:     int = 20,
    max_samples: int = 10000,
    save_dir:    str = '.',
    seed:        int = 42,
) -> list[dict]:
    """
    Load data, run the sweep, compute KL metrics at every step.
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    N     = min(dataset.size, max_samples)
    idx   = rng.choice(dataset.size, N, replace=False)
    batch = dataset.get_subset(idx)
    obs   = np.array(batch['observations'], dtype=np.float32)
    acts  = np.array(batch['actions'],      dtype=np.float32)

    x0_mm    = np.lib.format.open_memmap(x0_path, mode='r')
    x0_flat  = x0_mm.reshape(-1, x0_mm.shape[-1]) if x0_mm.ndim == 3 else x0_mm
    z_orig   = np.array(x0_flat[:N], dtype=np.float32)

    print(f"\n[Mode sweep] target_dim={target_dim}  mode_sign={mode_sign}  "
          f"n_steps={n_steps}  N={N}")

    steps   = mode_removal_sweep(obs, acts, z_orig,
                                 target_dim=target_dim, mode_sign=mode_sign,
                                 n_steps=n_steps, seed=seed)
    results = []

    for i, step in enumerate(steps):
        frac = step['remove_frac']
        print(f"\n  Step {i+1}/{len(steps)}: remove_frac={frac:.2f}  "
              f"n_remove={step['n_remove']}")

        if step['n_remove'] < 10:
            results.append(dict(
                remove_frac    = frac,
                kl_action      = 0.0,
                kl_noise       = 0.0,
                n_remove       = step['n_remove'],
                z_original     = np.array([]),
                z_hat_prime    = np.array([]),
                actions_subset = step['actions_subset'],
                actions_full   = step['actions_full'],
                skip           = True,
            ))
            continue

        kl_act = kl_kde(step['actions_full'], step['actions_subset'])
        print(f"    KL(action): {kl_act:.4f}")

        z_hat  = invert_subset_batched(agent, step['obs_subset'],
                                       step['actions_subset'])

        z_sub  = step['z_subset']
        kl_nois = kl_kde(z_sub.astype(np.float64),
                         z_hat.astype(np.float64)) if len(z_sub) >= 10 else 0.0
        print(f"    KL(noise):  {kl_nois:.4f}")

        results.append(dict(
            remove_frac    = frac,
            kl_action      = kl_act,
            kl_noise       = kl_nois,
            n_remove       = step['n_remove'],
            z_original     = z_sub,
            z_hat_prime    = z_hat,
            actions_subset = step['actions_subset'],
            actions_full   = step['actions_full'],
            skip           = False,
        ))

    return results


# ================================================================
# 4.  Non-Linearity Detection
# ================================================================

def fit_piecewise_linear(
    kl_action: np.ndarray,
    kl_noise:  np.ndarray,
) -> dict:
    """
    Fit: kl_noise = c0 for kl_action <= T*
         kl_noise = c0 + m*(kl_action - T*) for kl_action > T*
    Find T* by exhaustive grid search over candidate breakpoints.
    """
    n    = len(kl_action)
    kl_a = np.array(kl_action)
    kl_n = np.array(kl_noise)

    if n < 6:
        return dict(threshold=float(kl_a.mean()), flat_level=float(kl_n.mean()),
                    slope_before=0.0, slope_after=0.0, r2_improvement=0.0,
                    sse_piecewise=0.0, sse_baseline=0.0)

    best_sse = np.inf
    best_T   = kl_a[1]
    best_m   = 0.0
    best_b   = float(kl_n.mean())
    best_c   = float(kl_n.mean())

    for i in range(2, n - 2):
        T     = kl_a[i]
        left  = kl_a <= T
        right = kl_a >  T
        if left.sum() < 2 or right.sum() < 2:
            continue

        c_left = kl_n[left].mean()
        sse_l  = np.sum((kl_n[left] - c_left) ** 2)

        xl = kl_a[right];  yl = kl_n[right]
        m, b = np.polyfit(xl, yl, 1)
        sse_r = np.sum((yl - (m * xl + b)) ** 2)

        sse = sse_l + sse_r
        if sse < best_sse:
            best_sse = sse
            best_T   = T
            best_m   = m
            best_b   = b
            best_c   = c_left

    sse_baseline  = np.sum((kl_n - kl_n.mean()) ** 2)
    r2_improvement = 1.0 - best_sse / (sse_baseline + 1e-12)

    return dict(
        threshold      = float(best_T),
        flat_level     = float(best_c),
        slope_before   = 0.0,
        slope_after    = float(best_m),
        r2_improvement = float(r2_improvement),
        sse_piecewise  = float(best_sse),
        sse_baseline   = float(sse_baseline),
    )


def compute_linearity_score(
    kl_action: np.ndarray,
    kl_noise:  np.ndarray,
) -> dict:
    """
    Multiple quantifications of the KL(action) vs KL(noise) relationship.
    """
    kl_a = np.array(kl_action)
    kl_n = np.array(kl_noise)

    pearson_r,  pearson_p  = stats.pearsonr(kl_a, kl_n)
    spearman_r, spearman_p = stats.spearmanr(kl_a, kl_n)

    if len(kl_a) > 2:
        d_noise      = np.diff(kl_n)
        d_action     = np.diff(kl_a) + 1e-12
        local_slopes = d_noise / d_action
        max_slope    = float(np.max(np.abs(local_slopes)))
        mean_slope   = float(np.mean(np.abs(local_slopes)))
        slope_ratio  = max_slope / (mean_slope + 1e-12)
    else:
        max_slope   = 0.0
        mean_slope  = 0.0
        slope_ratio = 1.0

    pw = fit_piecewise_linear(kl_a, kl_n)

    return dict(
        pearson_r       = float(pearson_r),
        pearson_p       = float(pearson_p),
        spearman_r      = float(spearman_r),
        spearman_p      = float(spearman_p),
        max_local_slope = max_slope,
        mean_slope      = mean_slope,
        slope_ratio     = slope_ratio,
        piecewise       = pw,
    )


# ================================================================
# 5.  Visualisation
# ================================================================

def plot_original_vs_reduced(
    actions_all: np.ndarray,
    results:     list[dict],
    target_dim:  int,
    save_dir:    str = '.',
):
    valid = [r for r in results
             if not r.get('skip', False)
             and r['n_remove'] > 0
             and len(r['actions_subset']) > 0]

    if len(valid) == 0:
        print("  [warn] No valid steps for original_vs_reduced plot.")
        return

    n_panels = len(valid)
    ncols    = 5
    nrows    = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.8, nrows * 3.0),
                             sharey=False, squeeze=False)

    vals_all = actions_all[:, target_dim]
    bins     = np.linspace(vals_all.min() - 0.1,
                           vals_all.max() + 0.1, 50)

    for idx_panel, r in enumerate(valid):
        row = idx_panel // ncols
        col = idx_panel % ncols
        ax  = axes[row][col]

        removed_vals   = r['actions_subset'][:, target_dim]  # what was removed
        remaining_vals = r['actions_full'][:, target_dim]    # what is left

        # Original full set — grey reference, same every panel
        ax.hist(vals_all, bins=bins, density=True, alpha=0.20,
                color='#555', label='Original (full)')

        # Remaining set after removal — this should lose the +1 mode
        ax.hist(remaining_vals, bins=bins, density=True, alpha=0.55,
                color='#2a78d6', label=f'Remaining (n={len(remaining_vals)})')

        # Removed portion — the +1 mode chunk being taken out
        ax.hist(removed_vals, bins=bins, density=True, alpha=0.50,
                color='#d65f2a', label=f'Removed (n={r["n_remove"]})')

        pct = int(round(r['remove_frac'] * 100))
        ax.set_title(
            f"{pct}% of mode removed\n"
            f"KL_act={r['kl_action']:.3f}   KL_nois={r['kl_noise']:.4f}",
            fontsize=8,
        )
        ax.set_xlabel(f'Action dim {target_dim}', fontsize=8)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel('Density', fontsize=8)
        ax.legend(fontsize=6, loc='upper left')

    for idx_empty in range(n_panels, nrows * ncols):
        axes[idx_empty // ncols][idx_empty % ncols].set_visible(False)

    fig.suptitle(
        f"Action dim {target_dim}: original (grey) vs remaining (blue) vs removed (orange)\n"
        f"Blue = what the flow sees. Watch the +1 mode disappear from blue as panels progress.",
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    out = os.path.join(save_dir, 'original_vs_reduced.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  -> Figure saved: {out}")


def plot_action_distribution_shift(
    results:     list[dict],
    actions_all: np.ndarray,
    target_dim:  int,
    save_dir:    str = '.',
):
    """
    Five-panel snapshot: action distribution at 0%, 25%, 50%, 75%, 100% removal.
    """
    valid       = [r for r in results if not r.get('skip', False)]
    checkpoints = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=True)
    fig.suptitle(
        f"Action distribution in dim {target_dim} — five checkpoints",
        fontsize=12, fontweight='bold',
    )

    vals_all = actions_all[:, target_dim]
    bins     = np.linspace(vals_all.min() - 0.1,
                           vals_all.max() + 0.1, 50)

    for ax, target_frac in zip(axes, checkpoints):
        closest = min(valid, key=lambda r: abs(r['remove_frac'] - target_frac))

        ax.hist(vals_all, bins=bins, density=True, alpha=0.3,
                color='#888', label='Original')

        if closest['n_remove'] > 0 and len(closest['actions_subset']) > 0:
            removed_vals = closest['actions_subset'][:, target_dim]
            ax.hist(removed_vals, bins=bins, density=True, alpha=0.55,
                    color='#d65f2a',
                    label=f'Removed\n(n={closest["n_remove"]})')
        else:
            ax.hist(vals_all, bins=bins, density=True, alpha=0.7,
                    color='#2a78d6', label='Full (0% removed)')

        pct = int(round(target_frac * 100))
        ax.set_title(
            f"{pct}% removed\n"
            f"KL_act={closest['kl_action']:.3f}\n"
            f"KL_nois={closest['kl_noise']:.4f}",
        )
        ax.set_xlabel(f'dim {target_dim}')
        if ax is axes[0]:
            ax.set_ylabel('Density')
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(save_dir, 'action_distribution_shift.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  -> Figure saved: {out}")


def plot_mode_sweep(
    results:    list[dict],
    linearity:  dict,
    save_dir:   str = '.',
    target_dim: int = 0,
):
    """
    Four-panel non-linearity analysis figure.
    """
    valid = [r for r in results
             if not r.get('skip', False) and r['n_remove'] > 0]
    if len(valid) < 3:
        print("  Not enough valid results to plot mode_sweep_analysis.")
        return

    fracs      = np.array([r['remove_frac'] for r in valid])
    kl_actions = np.array([r['kl_action']   for r in valid])
    kl_noises  = np.array([r['kl_noise']    for r in valid])

    pw = linearity['piecewise']
    T  = pw['threshold']
    c0 = pw.get('flat_level', float(kl_noises.min()))
    m  = pw['slope_after']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Mode Removal Sweep — KL(action) vs KL(noise) Non-Linearity Analysis",
        fontsize=13, fontweight='bold',
    )

    # Panel A: dual-axis divergences vs removal fraction
    ax  = axes[0, 0]
    ax.plot(fracs, kl_actions, color='#2a78d6', marker='o', lw=2, ms=5,
            label='KL(action)')
    ax.set_xlabel('Fraction of +1 mode removed')
    ax.set_ylabel('KL(action)', color='#2a78d6')
    ax.tick_params(axis='y', labelcolor='#2a78d6')
    ax2 = ax.twinx()
    ax2.plot(fracs, kl_noises, color='#d65f2a', marker='s', lw=2, ms=5,
             linestyle='--', label='KL(noise)')
    ax2.set_ylabel('KL(noise)', color='#d65f2a')
    ax2.tick_params(axis='y', labelcolor='#d65f2a')
    ax2.spines['right'].set_visible(True)
    ax.set_title('(A) Divergences vs mode removal fraction')
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=9)

    # Panel B: KL(action) vs KL(noise) with piecewise fit
    ax = axes[0, 1]
    sc = ax.scatter(kl_actions, kl_noises, c=fracs, cmap='viridis',
                    s=70, zorder=3)
    plt.colorbar(sc, ax=ax, label='Remove fraction')
    x_flat = np.linspace(0, T, 50)
    x_rise = np.linspace(T, kl_actions.max(), 50)
    ax.plot(x_flat, np.full_like(x_flat, c0), 'k-',  lw=2.5,
            label='Flat regime')
    ax.plot(x_rise, c0 + m * (x_rise - T),   'r-',  lw=2.5,
            label=f'Rising (slope={m:.4f})')
    ax.axvline(T, color='red', lw=1.5, linestyle='--',
               label=f'T* = {T:.4f}')
    ax.set_xlabel('KL(action)')
    ax.set_ylabel('KL(noise)')
    ax.set_title(f'(B) KL(action) vs KL(noise)\n'
                 f'T* = {T:.4f}   R2 improvement = {pw["r2_improvement"]:.3f}')
    ax.legend(fontsize=8)

    # Panel C: local slope
    ax = axes[1, 0]
    if len(kl_actions) > 2:
        d_noise  = np.diff(kl_noises)
        d_action = np.diff(kl_actions) + 1e-12
        slopes   = d_noise / d_action
        bar_clrs = ['red' if abs(s) > linearity['mean_slope'] * 2
                    else '#2a78d6' for s in slopes]
        ax.bar(range(len(slopes)), slopes, color=bar_clrs,
               alpha=0.8, edgecolor='white')
        ax.axhline(linearity['mean_slope'],   color='k',   lw=1.5,
                   linestyle='--',
                   label=f'Mean = {linearity["mean_slope"]:.4f}')
        ax.axhline(linearity['mean_slope'] * 2, color='red', lw=1,
                   linestyle=':',  label='2x mean threshold')
        ax.set_xlabel('Step index')
        ax.set_ylabel('d(KL_noise) / d(KL_action)')
        ax.set_title(f'(C) Local slope — spike = non-linearity onset\n'
                     f'Max/mean ratio = {linearity["slope_ratio"]:.2f}')
        ax.legend(fontsize=8)

    # Panel D: summary text
    ax = axes[1, 1]
    ax.axis('off')
    rows = [
        ("NON-LINEARITY THRESHOLD", ""),
        (f"  T* (breakpoint)",          f"{T:.4f}"),
        (f"  Flat level c0",            f"{c0:.5f}"),
        (f"  Slope before T*",          "0  (flat by construction)"),
        (f"  Slope after T*",           f"{m:.5f}"),
        (f"  R2 improvement",           f"{pw['r2_improvement']:.4f}"),
        ("", ""),
        ("CORRELATION METRICS", ""),
        (f"  Pearson r",
         f"{linearity['pearson_r']:.4f}  (p={linearity['pearson_p']:.2e})"),
        (f"  Spearman r",
         f"{linearity['spearman_r']:.4f}  (p={linearity['spearman_p']:.2e})"),
        ("", ""),
        ("SLOPE ANALYSIS", ""),
        (f"  Mean local slope",         f"{linearity['mean_slope']:.5f}"),
        (f"  Max local slope",          f"{linearity['max_local_slope']:.5f}"),
        (f"  Slope ratio (max/mean)",   f"{linearity['slope_ratio']:.2f}"),
        ("", ""),
        ("INTERPRETATION", ""),
    ]

    if pw['r2_improvement'] > 0.3:
        interp_txt = "  Clear non-linearity above T*"
        interp_clr = 'red'
    elif pw['r2_improvement'] > 0.1:
        interp_txt = "  Weak non-linearity — borderline"
        interp_clr = 'darkorange'
    else:
        interp_txt = "  No non-linearity (flow normalises fully)"
        interp_clr = 'green'

    rows.append((interp_txt, ""))

    headers = {"NON-LINEARITY THRESHOLD", "CORRELATION METRICS",
               "SLOPE ANALYSIS", "INTERPRETATION"}
    y = 0.97
    for label, value in rows:
        if label in headers:
            ax.text(0.02, y, label, transform=ax.transAxes,
                    fontsize=10, fontweight='bold', color='#333')
        elif label == interp_txt:
            ax.text(0.02, y, label, transform=ax.transAxes,
                    fontsize=9.5, fontstyle='italic', color=interp_clr)
        else:
            ax.text(0.02, y, label, transform=ax.transAxes,
                    fontsize=9.5, color='#444')
            if value:
                ax.text(0.62, y, value, transform=ax.transAxes,
                        fontsize=9.5, color='#111')
        y -= 0.055

    ax.set_title('(D) Non-linearity summary')

    plt.tight_layout()
    out = os.path.join(save_dir, 'mode_sweep_analysis.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"\n  -> Figure saved: {out}")


# ================================================================
# 6.  CLI Entry Point
# ================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_dir',     required=True)
    p.add_argument('--x0_path',     required=True)
    p.add_argument('--env_name',    default=None)
    p.add_argument('--target_dim',  type=int, default=0)
    p.add_argument('--mode_sign',   type=int, default=1)
    p.add_argument('--n_steps',     type=int, default=20)
    p.add_argument('--max_samples', type=int, default=10000)
    p.add_argument('--save_dir',    default='analysis/mode_sweep')
    p.add_argument('--seed',        type=int, default=42)
    args = p.parse_args()

    # ── Load flags ─────────────────────────────────────────────
    import json as _json
    flags    = _json.load(open(os.path.join(args.exp_dir, 'flags.json')))
    env_name = args.env_name or flags['env_name']

    from envs.env_utils  import make_env_and_datasets
    from utils.datasets  import Dataset, ReplayBuffer
    from utils.flax_utils import restore_agent
    from agents          import agents as agent_classes
    import importlib.util

    # ── Load dataset ───────────────────────────────────────────
    _, _, train_dataset, _ = make_env_and_datasets(
        env_name, frame_stack=flags.get('frame_stack')
    )
    train_dataset = Dataset.create(**train_dataset)
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=train_dataset.size + 1
    )

    # ── Load agent config ──────────────────────────────────────
    spec    = importlib.util.spec_from_file_location(
        "fql_cfg",
        os.path.join(os.path.dirname(__file__), 'agents/fql.py'),
    )
    fql_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fql_mod)
    config  = fql_mod.get_config()

    ex    = train_dataset.sample(1)
    agent = agent_classes['fql'].create(
        flags.get('seed', 0), ex['observations'], ex['actions'], config
    )

    # ── Find and restore latest checkpoint ────────────────────
    epochs = []
    for nm in os.listdir(args.exp_dir):
        if nm.startswith('params_') and nm.endswith('.pkl'):
            try:
                epochs.append(int(nm.replace('params_', '').replace('.pkl', '')))
            except ValueError:
                pass
    epoch = max(epochs)
    agent = restore_agent(agent, args.exp_dir, epoch)
    print(f"Agent loaded from epoch {epoch}")

    # ── Load full action set for distribution plots ───────────
    # Must be defined BEFORE any plot calls
    idx_all     = np.random.RandomState(args.seed).choice(
        train_dataset.size,
        min(train_dataset.size, args.max_samples),
        replace=False,
    )
    batch_all   = train_dataset.get_subset(idx_all)
    actions_all = np.array(batch_all['actions'], dtype=np.float32)

    # ── Run sweep ──────────────────────────────────────────────
    results = run_mode_sweep(
        agent, train_dataset, args.x0_path,
        target_dim  = args.target_dim,
        mode_sign   = args.mode_sign,
        n_steps     = args.n_steps,
        max_samples = args.max_samples,
        save_dir    = args.save_dir,
        seed        = args.seed,
    )

    # ── Non-linearity metrics ──────────────────────────────────
    valid      = [r for r in results
                  if not r.get('skip', False) and r['n_remove'] > 0]
    kl_actions = np.array([r['kl_action'] for r in valid])
    kl_noises  = np.array([r['kl_noise']  for r in valid])
    linearity  = compute_linearity_score(kl_actions, kl_noises)

    print("\n" + "=" * 56)
    print("  Non-linearity analysis results")
    print("=" * 56)
    print(f"  Threshold T*:       {linearity['piecewise']['threshold']:.4f}")
    print(f"  Slope after T*:     {linearity['piecewise']['slope_after']:.5f}")
    print(f"  R2 improvement:     {linearity['piecewise']['r2_improvement']:.4f}")
    print(f"  Pearson r:          {linearity['pearson_r']:.4f}")
    print(f"  Slope ratio:        {linearity['slope_ratio']:.2f}")
    print("=" * 56)

    # ── All three plots (actions_all defined above) ────────────
    plot_mode_sweep(
        results   = results,
        linearity = linearity,
        save_dir  = args.save_dir,
        target_dim = args.target_dim,
    )

    plot_action_distribution_shift(
        results     = results,
        actions_all = actions_all,
        target_dim  = args.target_dim,
        save_dir    = args.save_dir,
    )

    plot_original_vs_reduced(
        actions_all = actions_all,
        results     = results,
        target_dim  = args.target_dim,
        save_dir    = args.save_dir,
    )

    # ── Save JSON metrics ──────────────────────────────────────
    def _ser(obj):
        if isinstance(obj, np.ndarray):            return obj.tolist()
        if isinstance(obj, (np.floating, float)):  return float(obj)
        if isinstance(obj, (np.integer, int)):     return int(obj)
        if isinstance(obj, dict):   return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list):   return [_ser(i) for i in obj]
        return obj

    out_json = os.path.join(args.save_dir, 'mode_sweep_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(_ser({
            'sweep': [
                {'remove_frac': r['remove_frac'],
                 'kl_action':   r['kl_action'],
                 'kl_noise':    r['kl_noise'],
                 'n_remove':    r['n_remove']}
                for r in results
            ],
            'linearity': linearity,
        }), f, indent=2)
    print(f"  -> Metrics saved: {out_json}")


if __name__ == '__main__':
    main()
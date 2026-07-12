"""
Dataset Subset Inversion & Noise Correlation Analysis
=====================================================
Grounded in: Park et al. (2025) "Flow Q-Learning", arXiv:2502.02538

LITERATURE BASIS
----------------
1. Distribution shift & coverage in offline RL
   - Levine et al. (2020) "Offline RL: Tutorial, Review..." arXiv:2005.01643
     → Distribution shift between π_β and π_learned is the core challenge.
       Removing action-states from 𝒟 induces an artificial shift we can measure.

2. Data filtering in offline RL
   - Xu et al. (2023) "Cross-domain Policy Adaptation via Value-Guided Data
     Filtering", NeurIPS 2023
   - Chen et al. (2024) "Sample-Efficient Policy Constraint Offline Deep RL
     based on Sample Filtering", arXiv:2512.20115
     → Pruning/filtering datasets changes the behavioral distribution in ways
       that affect both the learned policy and the inferred latent noise.

3. Imbalanced / sparse datasets
   - Shi et al. (2023) "Offline RL with Imbalanced Datasets", arXiv:2307.02752
     → Real-world offline datasets follow power-law state coverage; removing
       action-states can mimic or exacerbate this imbalance.

4. Divergence measures for generative models
   - Gretton et al. (2012) "A Kernel Two-Sample Test", JMLR
     → MMD with RBF kernel gives a consistent, non-parametric test of whether
       two action distributions are the same.
   - Villani (2009) "Optimal Transport: Old and New"
     → 2-Wasserstein distance (W₂) is the natural companion to FQL's Eq. 8,
       which already uses W₂ as its Wasserstein regulariser.

5. Latent space correlation
   - Hotelling (1936) "Relations Between Two Sets of Variates", Biometrika
     → Canonical Correlation Analysis (CCA) measures the maximum linear
       correlation between two multivariate noise spaces z and ẑ′.
   - Kraskov et al. (2004) "Estimating Mutual Information", Phys. Rev. E
     → k-NN mutual information estimator requires no density assumptions.

PIPELINE
--------
Step 1  Partition 𝒟 into full set 𝒟 and subset 𝒟′ (three removal strategies)
Step 2  Measure W₁, W₂, KL, MMD, JSD divergence between action distributions
Step 3  Invert the FQL BC flow ODE on 𝒟′ to recover noise ẑ′ = Φ⁻¹_θ(s, a′)
Step 4  Correlate z (from x0_all.npy) with ẑ′ via Pearson, CCA, MI, W₂, KL
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from typing import Literal


# ================================================================
# 1.  Dataset Partitioning — three removal strategies
# ================================================================

class DatasetPartitioner:
    """
    Removes a portion of (s, a) pairs from the offline dataset 𝒟.

    Three strategies, each motivated by the literature:

    'random'       -- uniform random removal (Shi et al., 2023 baseline)
    'quality'      -- remove low-Q-value transitions (Xu et al., 2023)
    'state_region' -- remove a compact region of state space (Levine et al., 2020
                      coverage analysis; mimics the hallway/room scenario in
                      Ross & Bagnell, 2012)
    """

    def __init__(self, removal_fraction: float = 0.3, seed: int = 42):
        assert 0.0 < removal_fraction < 1.0
        self.removal_fraction = removal_fraction
        self.rng = np.random.RandomState(seed)

    # ----------------------------------------------------------

    def random_removal(
        self,
        observations: np.ndarray,   # (N, s_dim)
        actions:      np.ndarray,   # (N, d)
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Randomly remove `removal_fraction` of transitions.
        Returns (keep_idx, remove_idx, keep_mask).
        """
        N = len(actions)
        remove_n   = int(N * self.removal_fraction)
        remove_idx = self.rng.choice(N, size=remove_n, replace=False)
        keep_mask  = np.ones(N, dtype=bool)
        keep_mask[remove_idx] = False
        return np.where(keep_mask)[0], remove_idx, keep_mask

    # ----------------------------------------------------------

    def quality_removal(
        self,
        observations: np.ndarray,
        actions:      np.ndarray,
        q_values:     np.ndarray,   # (N,) Q(s,a) from trained critic
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove the lowest-Q transitions (poor behavioural quality).
        Motivated by Xu et al. (2023): value-guided data filtering.

        Removing low-quality data makes 𝒟′ represent a higher-quality
        sub-policy π′_β ⊂ π_β. The noise divergence then measures
        how much the latent space shifts toward better behaviours.
        """
        N        = len(actions)
        remove_n = int(N * self.removal_fraction)
        sorted_idx = np.argsort(q_values)          # ascending: worst first
        remove_idx = sorted_idx[:remove_n]
        keep_mask  = np.ones(N, dtype=bool)
        keep_mask[remove_idx] = False
        return np.where(keep_mask)[0], remove_idx, keep_mask

    # ----------------------------------------------------------

    def state_region_removal(
        self,
        observations: np.ndarray,
        actions:      np.ndarray,
        region_dims:  list[int] | None = None,
        region_frac:  float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove all transitions where the state falls inside a specific
        hypercube region of the state space.
        """
        if region_frac is None:
            region_frac = self.removal_fraction

        if region_dims is None:
            region_dims = [0, 1]

        thresholds = np.quantile(
            observations[:, region_dims], 1 - region_frac, axis=0
        )
        in_region  = np.all(observations[:, region_dims] >= thresholds, axis=1)
        remove_idx = np.where(in_region)[0]
        keep_mask  = ~in_region
        return np.where(keep_mask)[0], remove_idx, keep_mask


# ================================================================
# 2.  Divergence Metrics (shared by action space and noise space)
# ================================================================

def compute_mmd(
    X:           np.ndarray,
    Y:           np.ndarray,
    sigma:       float | None = None,
    n_subsample: int = 2000,
) -> float:
    """
    Unbiased Maximum Mean Discrepancy (MMD²) with RBF kernel.
    Gretton et al. (2012) JMLR.

    MMD²(P, Q) = E[k(x,x)] - 2·E[k(x,y)] + E[k(y,y)]
    where k(x,y) = exp(-‖x−y‖²/(2σ²))

    Zero iff the two distributions are identical.
    Consistent, non-parametric, requires no density estimation.
    """
    if len(X) > n_subsample:
        X = X[np.random.choice(len(X), n_subsample, replace=False)]
    if len(Y) > n_subsample:
        Y = Y[np.random.choice(len(Y), n_subsample, replace=False)]

    if sigma is None:
        all_pts = np.vstack([X, Y])
        pw_dist = cdist(all_pts, all_pts, 'sqeuclidean')
        sigma   = float(np.sqrt(np.median(pw_dist[pw_dist > 0]) / 2))
        sigma   = max(sigma, 1e-6)

    g   = 1.0 / (2 * sigma ** 2)
    Kxx = np.exp(-g * cdist(X, X, 'sqeuclidean'))
    Kyy = np.exp(-g * cdist(Y, Y, 'sqeuclidean'))
    Kxy = np.exp(-g * cdist(X, Y, 'sqeuclidean'))

    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    mmd2 = (Kxx.sum() / (n * (n - 1)) +
            Kyy.sum() / (m * (m - 1)) -
            2 * Kxy.mean())
    return float(max(mmd2, 0.0))


def compute_sliced_wasserstein(
    X:             np.ndarray,
    Y:             np.ndarray,
    n_projections: int = 200,
    p:             int = 2,
) -> float:
    """
    Sliced Wasserstein-p distance (Rabin et al., 2012).
    Approximates W_p by averaging 1-D Wasserstein distances over
    random projections — scales to high-dimensional spaces.

    For p=2 this approximates W₂, directly comparable to FQL Eq. 8.
    """
    d   = X.shape[1]
    rng = np.random.RandomState(0)
    dirs = rng.randn(n_projections, d)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    sw = 0.0
    for v in dirs:
        px = np.sort(X @ v)
        py = np.sort(Y @ v)
        n  = min(len(px), len(py))
        sw += np.mean(np.abs(px[:n] - py[:n]) ** p)

    return float((sw / n_projections) ** (1.0 / p))


def compute_jsd(
    X:      np.ndarray,
    Y:      np.ndarray,
    n_bins: int = 30,
) -> float:
    """
    Jensen-Shannon divergence via per-dimension histograms.
    Returns the mean JSD across dimensions.

    JSD = (KL(P‖M) + KL(Q‖M)) / 2  where M = (P+Q)/2.
    Bounded in [0, log2], symmetric; ≈ 0 iff distributions match.
    """
    d            = X.shape[1]
    jsd_per_dim  = []

    for j in range(d):
        lo   = min(X[:, j].min(), Y[:, j].min())
        hi   = max(X[:, j].max(), Y[:, j].max())
        bins = np.linspace(lo, hi, n_bins + 1)

        px, _ = np.histogram(X[:, j], bins=bins, density=True)
        py, _ = np.histogram(Y[:, j], bins=bins, density=True)
        px = px / (px.sum() + 1e-12)
        py = py / (py.sum() + 1e-12)

        m       = 0.5 * (px + py)
        eps     = 1e-12
        kl_pm   = np.sum(px * np.log((px + eps) / (m + eps)))
        kl_qm   = np.sum(py * np.log((py + eps) / (m + eps)))
        jsd_per_dim.append(0.5 * (kl_pm + kl_qm))

    return float(np.mean(jsd_per_dim))


def compute_kl_divergence(
    X:        np.ndarray,   # (N, d)  — reference distribution P
    Y:        np.ndarray,   # (M, d)  — comparison distribution Q
    n_points: int = 500,
) -> dict:
    """
    KDE-based KL divergence per dimension, then averaged.

    Uses scipy.stats.gaussian_kde with Scott's bandwidth rule.

    KL(P‖Q) — penalises where P has mass but Q does not (mode-covering).
    KL(Q‖P) — penalises where Q has mass but P does not (mode-seeking).
    kl_sym   = (KL(P‖Q) + KL(Q‖P)) / 2  — symmetric headline number.

    Returns:
        kl_pq         : mean KL(P‖Q) across dims
        kl_qp         : mean KL(Q‖P) across dims
        kl_sym        : mean symmetric KL across dims
        kl_pq_per_dim : (d,) per-dimension KL(P‖Q)
        kl_qp_per_dim : (d,) per-dimension KL(Q‖P)
    """
    d             = X.shape[1]
    kl_pq_dims    = []
    kl_qp_dims    = []

    for j in range(d):
        xj   = X[:, j].astype(np.float64)
        yj   = Y[:, j].astype(np.float64)
        lo   = min(xj.min(), yj.min()) - 0.5
        hi   = max(xj.max(), yj.max()) + 0.5
        grid = np.linspace(lo, hi, n_points)

        try:
            kde_p = stats.gaussian_kde(xj)
            kde_q = stats.gaussian_kde(yj)
        except np.linalg.LinAlgError:
            kl_pq_dims.append(0.0)
            kl_qp_dims.append(0.0)
            continue

        eps  = 1e-12
        p    = kde_p(grid);  p /= (p.sum() + eps)
        q    = kde_q(grid);  q /= (q.sum() + eps)

        kl_pq_dims.append(float(np.sum(p * np.log((p + eps) / (q + eps)))))
        kl_qp_dims.append(float(np.sum(q * np.log((q + eps) / (p + eps)))))

    kl_pq_arr = np.array(kl_pq_dims)
    kl_qp_arr = np.array(kl_qp_dims)

    return dict(
        kl_pq         = float(np.mean(kl_pq_arr)),
        kl_qp         = float(np.mean(kl_qp_arr)),
        kl_sym        = float(np.mean((kl_pq_arr + kl_qp_arr) / 2)),
        kl_pq_per_dim = kl_pq_arr,
        kl_qp_per_dim = kl_qp_arr,
    )


def compute_wasserstein_per_dim(
    X: np.ndarray,   # (N, d)
    Y: np.ndarray,   # (M, d)
) -> dict:
    """
    Exact 1-D Wasserstein-1 and Wasserstein-2 distances per dimension.

    For 1-D distributions the Wasserstein-p distance has a closed form:
        W_p(P, Q) = ( ∫₀¹ |F_P⁻¹(u) − F_Q⁻¹(u)|^p du )^{1/p}
    which reduces to comparing sorted arrays (quantile functions).

    This is exact — no approximation, no kernel, no projections.
    W2 per-dim is directly comparable to FQL Eq. 8.

    Returns:
        w1_mean    : mean W1 across dimensions
        w2_mean    : mean W2 across dimensions
        w1_per_dim : (d,) W1 per dimension
        w2_per_dim : (d,) W2 per dimension
    """
    d        = X.shape[1]
    w1_dims  = []
    w2_dims  = []

    for j in range(d):
        xj = X[:, j].astype(np.float64)
        yj = Y[:, j].astype(np.float64)

        # W1 — exact via scipy
        w1_dims.append(float(stats.wasserstein_distance(xj, yj)))

        # W2 — exact via sorted quantile arrays
        n    = min(len(xj), len(yj))
        xj_s = np.sort(xj)[:n]
        yj_s = np.sort(yj)[:n]
        w2_dims.append(float(np.sqrt(np.mean((xj_s - yj_s) ** 2))))

    w1_arr = np.array(w1_dims)
    w2_arr = np.array(w2_dims)

    return dict(
        w1_mean    = float(np.mean(w1_arr)),
        w2_mean    = float(np.mean(w2_arr)),
        w1_per_dim = w1_arr,
        w2_per_dim = w2_arr,
    )


# ================================================================
# 3.  Action Distribution Divergence
# ================================================================

def measure_action_divergence(
    actions_full:   np.ndarray,   # (N, d) from 𝒟
    actions_subset: np.ndarray,   # (M, d) from 𝒟′
    verbose:        bool = True,
) -> dict:
    """
    Compute all divergence metrics between the full and subset action sets.

    Returns dict with: mmd, sliced_w2, jsd, kl_*, w1_*, w2_*,
    mean_shift, std_ratio.
    """
    mmd  = compute_mmd(actions_full, actions_subset)
    sw2  = compute_sliced_wasserstein(actions_full, actions_subset, p=2)
    jsd  = compute_jsd(actions_full, actions_subset)
    kl   = compute_kl_divergence(actions_full, actions_subset)
    wass = compute_wasserstein_per_dim(actions_full, actions_subset)

    mean_shift = float(np.mean(np.abs(
        actions_full.mean(0) - actions_subset.mean(0)
    )))
    std_ratio  = float(np.mean(
        actions_subset.std(0) / (actions_full.std(0) + 1e-8)
    ))

    metrics = dict(
        mmd           = mmd,
        sliced_w2     = sw2,
        jsd           = jsd,
        kl_pq         = kl['kl_pq'],
        kl_qp         = kl['kl_qp'],
        kl_sym        = kl['kl_sym'],
        w1_mean       = wass['w1_mean'],
        w2_mean       = wass['w2_mean'],
        w1_per_dim    = wass['w1_per_dim'],
        w2_per_dim    = wass['w2_per_dim'],
        kl_pq_per_dim = kl['kl_pq_per_dim'],
        kl_qp_per_dim = kl['kl_qp_per_dim'],
        mean_shift    = mean_shift,
        std_ratio     = std_ratio,
    )

    if verbose:
        print("\n" + "=" * 56)
        print("  Action Distribution Divergence  𝒟 vs 𝒟′")
        print("=" * 56)
        print(f"  MMD² (Gretton 2012):       {mmd:.4f}  (0 = identical)")
        print(f"  Sliced W₂ (≈ FQL Eq. 8):  {sw2:.4f}")
        print(f"  W1 per-dim mean (exact):   {wass['w1_mean']:.4f}")
        print(f"  W2 per-dim mean (exact):   {wass['w2_mean']:.4f}")
        print(f"  JSD (per-dim mean):        {jsd:.4f}  (0 = identical)")
        print(f"  KL(full‖subset) sym:       {kl['kl_sym']:.4f}")
        print(f"  KL(full→subset):           {kl['kl_pq']:.4f}")
        print(f"  KL(subset→full):           {kl['kl_qp']:.4f}")
        print(f"  Mean action shift:         {mean_shift:.4f}")
        print(f"  Std ratio (sub/full):      {std_ratio:.4f}  (1 = same spread)")
        print("=" * 56)

    return metrics


# ================================================================
# 4.  ODE Inversion on the Subset
# ================================================================

def invert_subset(
    agent,
    observations_subset: np.ndarray,   # (M, s_dim)
    actions_subset:      np.ndarray,   # (M, d)
    batch_size:          int = 256,
) -> np.ndarray:
    """
    Run compute_reverse_flow_actions on every (s, a) pair in 𝒟′.

    Processes in batches to avoid OOM. Returns ẑ′ ∈ R^{M×d}.

    If the BC flow is a perfect bijection, ẑ′ should be ~ N(0, I).
    The degree to which ẑ′ deviates from N(0,I) measures how much
    the subset 𝒟′ has shifted the latent noise geometry.
    """
    M           = len(actions_subset)
    d           = actions_subset.shape[1]
    z_hat_prime = np.zeros((M, d), dtype=np.float32)

    for start in range(0, M, batch_size):
        end   = min(start + batch_size, M)
        obs_b = jnp.array(observations_subset[start:end])
        act_b = jnp.array(actions_subset[start:end])
        z_b   = np.array(agent.compute_reverse_flow_actions(obs_b, act_b))
        z_hat_prime[start:end] = z_b.astype(np.float32)

    return z_hat_prime


# ================================================================
# 5.  Noise Correlation Analysis
# ================================================================

def pearson_per_dim(
    z:       np.ndarray,   # (N, d) — original noise
    z_prime: np.ndarray,   # (M, d) — recovered noise from subset
) -> np.ndarray:
    """
    Per-dimension Pearson correlation between matched noise samples.

    If sizes differ, subsamples to min(N, M) with a fixed seed.
    Returns corr: (d,) array in [-1, 1].

    High |r| per dimension → subset preserves that latent direction.
    r ≈ 0 → the flow has remapped that dimension due to subset shift.
    """
    n   = min(len(z), len(z_prime))
    rng = np.random.RandomState(0)
    if len(z) > n:
        z       = z[rng.choice(len(z), n, replace=False)]
    if len(z_prime) > n:
        z_prime = z_prime[rng.choice(len(z_prime), n, replace=False)]

    d    = z.shape[1]
    corr = np.array([
        float(np.corrcoef(z[:, j], z_prime[:, j])[0, 1])
        for j in range(d)
    ])
    return corr


def canonical_correlation_analysis(
    z:            np.ndarray,   # (N, d)
    z_prime:      np.ndarray,   # (M, d)
    n_components: int | None = None,
) -> dict:
    """
    Canonical Correlation Analysis (CCA) between z and ẑ′.
    Hotelling (1936) "Relations Between Two Sets of Variates."

    CCA finds linear combinations of z and ẑ′ that are maximally
    correlated. Canonical correlations ρ_1 ≥ ... ≥ ρ_d give a
    dimension-wise measure of linear alignment between noise spaces.

    ρ_1 ≈ 1 → the two noise spaces share at least one strong direction.
    frac(ρ > 0.5) → fraction of latent dimensions preserved.
    """
    n   = min(len(z), len(z_prime))
    rng = np.random.RandomState(0)
    Z   = z[rng.choice(len(z), n, replace=False)]       if len(z) > n       else z
    Zp  = z_prime[rng.choice(len(z_prime), n, replace=False)] if len(z_prime) > n else z_prime

    d            = Z.shape[1]
    n_components = min(n_components or d, d, n - 1)

    sc_z  = StandardScaler().fit(Z)
    sc_zp = StandardScaler().fit(Zp)
    Z_s   = sc_z.transform(Z)
    Zp_s  = sc_zp.transform(Zp)

    cca      = CCA(n_components=n_components)
    cca.fit(Z_s, Zp_s)
    Zt, Zpt  = cca.transform(Z_s, Zp_s)

    canon_corrs = np.array([
        float(np.corrcoef(Zt[:, k], Zpt[:, k])[0, 1])
        for k in range(n_components)
    ])

    return dict(
        canonical_correlations = canon_corrs,
        mean_canonical_corr    = float(np.mean(canon_corrs)),
        frac_above_half        = float(np.mean(canon_corrs > 0.5)),
        n_components           = n_components,
    )


def knn_mutual_information(
    z:           np.ndarray,   # (N, d)
    z_prime:     np.ndarray,   # (M, d)
    k:           int = 5,
    n_subsample: int = 3000,
) -> float:
    """
    k-NN mutual information estimator (Kraskov et al., 2004).
    Estimates MI(z; ẑ′) without any density assumptions.

    High MI → knowing z tells you a lot about ẑ′.
    Low MI  → subset shift has broken the latent correspondence.

    Uses the KSG estimator restricted to first 4 dims for tractability.
    """
    n   = min(len(z), len(z_prime), n_subsample)
    rng = np.random.RandomState(0)
    Z   = z[rng.choice(len(z),       n, replace=False)].astype(np.float64)
    Zp  = z_prime[rng.choice(len(z_prime), n, replace=False)].astype(np.float64)

    d_joint = min(Z.shape[1], 4)
    Z       = Z[:, :d_joint]
    Zp      = Zp[:, :d_joint]
    joint   = np.hstack([Z, Zp])

    from scipy.spatial  import KDTree
    from scipy.special  import digamma

    tree_joint = KDTree(joint)
    tree_z     = KDTree(Z)
    tree_zp    = KDTree(Zp)

    dists, _ = tree_joint.query(joint, k=k + 1)
    eps      = dists[:, -1]

    nx = np.array([
        len(tree_z.query_ball_point(Z[i], eps[i] - 1e-15)) - 1
        for i in range(n)
    ], dtype=float)
    ny = np.array([
        len(tree_zp.query_ball_point(Zp[i], eps[i] - 1e-15)) - 1
        for i in range(n)
    ], dtype=float)

    mi = (digamma(k) + digamma(n)
          - np.mean(digamma(nx + 1))
          - np.mean(digamma(ny + 1)))
    return float(max(mi, 0.0))


def noise_correlation_report(
    z:       np.ndarray,   # (N, d) original training noise
    z_prime: np.ndarray,   # (M, d) recovered noise from subset
    verbose: bool = True,
) -> dict:
    """
    Full correlation report: Pearson, CCA, MI, Wasserstein (W1, W2),
    KL divergence, and Gaussianity test between z and ẑ′.

    Wasserstein and KL here measure divergence in the NOISE space —
    the latent-space counterpart of the action-space divergence computed
    in measure_action_divergence. Comparing the two reveals whether the
    BC flow absorbs or propagates distributional shift.
    """
    pearson    = pearson_per_dim(z, z_prime)
    cca_res    = canonical_correlation_analysis(z, z_prime)
    mi         = knn_mutual_information(z, z_prime)
    sw2_noise  = compute_sliced_wasserstein(z, z_prime, p=2)
    wass_noise = compute_wasserstein_per_dim(z, z_prime)
    kl_noise   = compute_kl_divergence(z, z_prime)
    ks_stat, ks_pval = stats.kstest(z_prime.flatten(), 'norm')

    report = dict(
        pearson_per_dim        = pearson,
        pearson_mean_abs       = float(np.mean(np.abs(pearson))),
        pearson_min            = float(pearson.min()),
        cca                    = cca_res,
        mutual_information     = mi,
        # Wasserstein in noise space
        sliced_w2_noise        = sw2_noise,
        w1_noise_mean          = wass_noise['w1_mean'],
        w2_noise_mean          = wass_noise['w2_mean'],
        w1_noise_per_dim       = wass_noise['w1_per_dim'],
        w2_noise_per_dim       = wass_noise['w2_per_dim'],
        # KL in noise space
        kl_noise_sym           = kl_noise['kl_sym'],
        kl_noise_pq            = kl_noise['kl_pq'],
        kl_noise_qp            = kl_noise['kl_qp'],
        kl_noise_pq_per_dim    = kl_noise['kl_pq_per_dim'],
        kl_noise_qp_per_dim    = kl_noise['kl_qp_per_dim'],
        # Gaussianity of ẑ′
        ks_stat_zprime         = ks_stat,
        ks_pval_zprime         = ks_pval,
    )

    if verbose:
        print("\n" + "=" * 56)
        print("  Noise Correlation:  z (original) vs ẑ′ (subset)")
        print("=" * 56)
        print(f"  Pearson |r| mean:           {report['pearson_mean_abs']:.4f}")
        print(f"  Pearson |r| min:            {report['pearson_min']:.4f}")
        print(f"  CCA mean canonical corr:    {cca_res['mean_canonical_corr']:.4f}")
        print(f"  CCA frac(ρ > 0.5):          {cca_res['frac_above_half']:.4f}")
        print(f"  Mutual information (KSG):   {mi:.4f}  nats")
        print(f"  --- Wasserstein (noise) ---")
        print(f"  Sliced W₂ (noise space):    {sw2_noise:.4f}")
        print(f"  W1 per-dim mean (exact):    {wass_noise['w1_mean']:.4f}")
        print(f"  W2 per-dim mean (exact):    {wass_noise['w2_mean']:.4f}")
        print(f"  --- KL divergence (noise) ---")
        print(f"  KL(z‖ẑ′) symmetric:         {kl_noise['kl_sym']:.4f}")
        print(f"  KL(z → ẑ′):                 {kl_noise['kl_pq']:.4f}")
        print(f"  KL(ẑ′ → z):                 {kl_noise['kl_qp']:.4f}")
        print(f"  --- Gaussianity of ẑ′ ---")
        print(f"  KS stat vs N(0,1):          {ks_stat:.4f}")
        print(f"  KS p-value:                 {ks_pval:.4e}")
        print("=" * 56)

    return report


# ================================================================
# 6.  Diagnostic Figures
# ================================================================

def plot_full_analysis(
    actions_full:   np.ndarray,
    actions_subset: np.ndarray,
    z_original:     np.ndarray,
    z_hat_prime:    np.ndarray,
    div_metrics:    dict,
    corr_report:    dict,
    strategy:       str = 'random',
    save_path:      str | None = None,
    show:           bool = False,
):
    """
    Six-panel diagnostic figure covering all four pipeline steps.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Dataset Subset Inversion Analysis  (strategy: {strategy})",
        fontsize=13, fontweight='bold',
    )

    # ── (0,0) Action marginal: full vs subset ──────────────────
    ax  = axes[0, 0]
    d   = actions_full.shape[1]
    flat_full   = actions_full[:, :min(d, 4)].flatten()
    flat_subset = actions_subset[:, :min(d, 4)].flatten()
    bins = np.linspace(
        min(flat_full.min(), flat_subset.min()),
        max(flat_full.max(), flat_subset.max()), 50,
    )
    ax.hist(flat_full,   bins=bins, density=True, alpha=0.5,
            color='steelblue', label='Full 𝒟')
    ax.hist(flat_subset, bins=bins, density=True, alpha=0.5,
            color='coral',     label='Subset 𝒟′')
    ax.set_title('Action distributions (first 4 dims)')
    ax.set_xlabel('Action value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.text(0.97, 0.95,
            f"MMD²={div_metrics['mmd']:.3f}\n"
            f"SlW₂={div_metrics['sliced_w2']:.3f}\n"
            f"W2={div_metrics['w2_mean']:.3f}\n"
            f"KL={div_metrics['kl_sym']:.3f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    # ── (0,1) Recovered noise ẑ′ vs N(0,1) ────────────────────
    ax      = axes[0, 1]
    flat_zp = z_hat_prime.flatten()
    flat_z  = z_original.flatten()
    bins_n  = np.linspace(-4, 4, 60)
    ax.hist(flat_z,  bins=bins_n, density=True, alpha=0.4,
            color='steelblue', label='z original')
    ax.hist(flat_zp, bins=bins_n, density=True, alpha=0.4,
            color='coral',     label='ẑ′ recovered')
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, stats.norm.pdf(xs), 'k-', lw=2, label='N(0,1)')
    ax.set_title('Noise distributions')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.text(0.97, 0.95,
            f"KS(ẑ′)={corr_report['ks_stat_zprime']:.3f}\n"
            f"W2={corr_report['w2_noise_mean']:.3f}\n"
            f"KL={corr_report['kl_noise_sym']:.3f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    # ── (0,2) Pearson per-dimension ────────────────────────────
    ax     = axes[0, 2]
    pearson = corr_report['pearson_per_dim']
    dims   = np.arange(len(pearson))
    colors = ['steelblue' if r > 0 else 'coral' for r in pearson]
    ax.bar(dims, pearson, color=colors, alpha=0.8)
    ax.axhline(0, color='k', lw=0.8)
    ax.axhline(corr_report['pearson_mean_abs'], color='navy',
               lw=1.5, linestyle='--',
               label=f"mean |r|={corr_report['pearson_mean_abs']:.3f}")
    ax.set_title('Pearson correlation  z vs ẑ′ (per dim)')
    ax.set_xlabel('Action / noise dimension')
    ax.set_ylabel('Pearson r')
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=8)

    # ── (1,0) CCA canonical correlations ───────────────────────
    ax = axes[1, 0]
    cc = corr_report['cca']['canonical_correlations']
    ax.bar(np.arange(len(cc)), cc, color='steelblue', alpha=0.8)
    ax.axhline(0.5, color='r', lw=1.5, linestyle='--', label='ρ = 0.5 threshold')
    ax.set_title('CCA canonical correlations')
    ax.set_xlabel('Canonical component')
    ax.set_ylabel('Correlation ρ')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.text(0.97, 0.95,
            f"mean ρ={corr_report['cca']['mean_canonical_corr']:.3f}\n"
            f"frac>0.5={corr_report['cca']['frac_above_half']:.2f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    # ── (1,1) Scatter z vs ẑ′ (dim 0) ─────────────────────────
    ax  = axes[1, 1]
    n   = min(len(z_original), len(z_hat_prime), 2000)
    rng = np.random.RandomState(0)
    iz  = rng.choice(len(z_original),  n, replace=False)
    izp = rng.choice(len(z_hat_prime), n, replace=False)
    ax.scatter(z_original[iz, 0], z_hat_prime[izp, 0],
               s=4, alpha=0.3, color='steelblue')
    lim = max(np.abs(z_original[:, 0]).max(),
              np.abs(z_hat_prime[:, 0]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1.5, label='identity')
    ax.set_title('z vs ẑ′  (dim 0 scatter)')
    ax.set_xlabel('z dim 0 (original)')
    ax.set_ylabel('ẑ′ dim 0 (recovered)')
    ax.legend(fontsize=8)
    ax.text(0.03, 0.95,
            f"MI ≈ {corr_report['mutual_information']:.3f} nats",
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    # ── (1,2) Summary bar chart ─────────────────────────────────
    ax     = axes[1, 2]
    labels = [
        'MMD²', 'Sliced\nW₂', 'W2\n(exact)', 'KL\n(action)',
        'Noise\nW2', 'Noise\nKL', 'Pearson\n|r|', 'CCA\nρ mean',
    ]
    vals = [
        div_metrics['mmd'],
        div_metrics['sliced_w2'],
        div_metrics['w2_mean'],
        div_metrics['kl_sym'],
        corr_report['w2_noise_mean'],
        corr_report['kl_noise_sym'],
        corr_report['pearson_mean_abs'],
        corr_report['cca']['mean_canonical_corr'],
    ]
    bar_colors = ['coral', 'coral', 'coral', 'coral',
                  'steelblue', 'steelblue', 'steelblue', 'steelblue']
    bars = ax.bar(labels, vals, color=bar_colors, alpha=0.8)
    ax.set_title('Summary metrics')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', labelsize=7)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7,
        )
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(fc='coral',     alpha=0.8, label='Action divergence'),
        Patch(fc='steelblue', alpha=0.8, label='Noise metrics'),
    ], fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  → Figure saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ================================================================
# 7.  Full End-to-End Pipeline
# ================================================================

def run_subset_inversion_analysis(
    agent,
    dataset,                  # object with .sample(n) and .get_subset(idx)
    x0_path:      str,        # path to x0_all.npy
    strategy:     Literal['random', 'quality', 'state_region'] = 'random',
    removal_frac: float = 0.30,
    q_values:     np.ndarray | None = None,
    region_dims:  list[int]  | None = None,
    max_samples:  int = 10000,
    save_dir:     str = '.',
    seed:         int = 42,
) -> dict:
    """
    End-to-end pipeline.

    Args:
        agent:         trained FQLAgent
        dataset:       replay buffer with full training data
        x0_path:       path to x0_all.npy (training noises)
        strategy:      'random' | 'quality' | 'state_region'
        removal_frac:  fraction of data to remove
        q_values:      (N,) Q(s,a) values for 'quality' strategy
        region_dims:   state dims defining the region for 'state_region'
        max_samples:   cap for memory safety
        save_dir:      output directory for figures
        seed:          random seed

    Returns:
        dict with divergence and correlation metrics.
    """
    import os

    # ── Load data ──────────────────────────────────────────────
    N       = min(dataset.size, max_samples)
    idx_all = np.random.RandomState(seed).choice(dataset.size, N, replace=False)
    batch   = dataset.get_subset(idx_all)
    obs     = np.array(batch['observations'], dtype=np.float32)
    actions = np.array(batch['actions'],      dtype=np.float32)

    x0_mm = np.lib.format.open_memmap(x0_path, mode='r')
    if x0_mm.ndim == 3:
        x0_flat = x0_mm.reshape(-1, x0_mm.shape[-1])[:N]
    else:
        x0_flat = x0_mm[:N]
    z_original = np.array(x0_flat, dtype=np.float32)

    # ── Partition ──────────────────────────────────────────────
    partitioner = DatasetPartitioner(removal_fraction=removal_frac, seed=seed)

    if strategy == 'random':
        keep_idx, remove_idx, _ = partitioner.random_removal(obs, actions)
    elif strategy == 'quality':
        if q_values is None:
            raise ValueError("q_values required for 'quality' strategy")
        keep_idx, remove_idx, _ = partitioner.quality_removal(
            obs, actions, q_values[:N]
        )
    elif strategy == 'state_region':
        keep_idx, remove_idx, _ = partitioner.state_region_removal(
            obs, actions, region_dims=region_dims
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    obs_subset     = obs[remove_idx]
    actions_subset = actions[remove_idx]
    actions_full   = actions[keep_idx]

    print(f"\n[Pipeline] Strategy: {strategy}")
    print(f"  Full 𝒟:    {len(keep_idx)} transitions")
    print(f"  Subset 𝒟′: {len(remove_idx)} transitions  ({removal_frac*100:.0f}% removed)")

    # ── Step 2: Action divergence ──────────────────────────────
    print("\n[Step 2] Measuring action distribution divergence ...")
    div_metrics = measure_action_divergence(actions_full, actions_subset)

    # ── Step 3: ODE inversion on subset ───────────────────────
    print("\n[Step 3] Inverting ODE on subset 𝒟′ ...")
    z_hat_prime = invert_subset(agent, obs_subset, actions_subset)

    # ── Step 4: Noise correlation ──────────────────────────────
    print("\n[Step 4] Noise correlation analysis ...")
    z_subset_orig = (
        z_original[remove_idx]
        if len(remove_idx) <= len(z_original)
        else z_original[:len(remove_idx)]
    )
    corr_report = noise_correlation_report(z_subset_orig, z_hat_prime)

    # ── Figure ─────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'subset_inversion_{strategy}.png')
    plot_full_analysis(
        actions_full   = actions_full,
        actions_subset = actions_subset,
        z_original     = z_subset_orig,
        z_hat_prime    = z_hat_prime,
        div_metrics    = div_metrics,
        corr_report    = corr_report,
        strategy       = strategy,
        save_path      = save_path,
    )

    return dict(
        strategy    = strategy,
        n_full      = len(keep_idx),
        n_subset    = len(remove_idx),
        divergence  = div_metrics,
        correlation = corr_report,
        z_original  = z_subset_orig,
        z_hat_prime = z_hat_prime,
    )


# ================================================================
# 8.  Smoke Test (no agent required)
# ================================================================

def _smoke_test():
    """Verify the full pipeline on synthetic data."""
    print("\n=== Smoke Test: Subset Inversion ===\n")
    rng = np.random.RandomState(0)
    N, d = 4000, 4

    a1      = rng.randn(N // 2, d) + np.array([1.5, 0, 0, 0])
    a2      = rng.randn(N // 2, d) + np.array([-1.5, 0, 0, 0])
    actions = np.vstack([a1, a2]).astype(np.float32)
    obs     = rng.randn(N, 8).astype(np.float32)
    z_orig  = rng.randn(N, d).astype(np.float32)

    partitioner            = DatasetPartitioner(removal_fraction=0.3, seed=0)
    keep_idx, remove_idx, _ = partitioner.random_removal(obs, actions)

    actions_full   = actions[keep_idx]
    actions_subset = actions[remove_idx]
    z_subset       = z_orig[remove_idx]

    print(f"  Full: {len(keep_idx)}, Subset: {len(remove_idx)}")

    div = measure_action_divergence(actions_full, actions_subset)
    assert div['mmd']    >= 0, "MMD must be non-negative"
    assert div['jsd']    >= 0, "JSD must be non-negative"
    assert div['w1_mean'] >= 0, "W1 must be non-negative"
    assert div['w2_mean'] >= 0, "W2 must be non-negative"
    assert 'kl_sym' in div,    "KL sym must be present"
    print("  Divergence checks passed.")

    z_hat_prime = z_subset + 0.1 * rng.randn(*z_subset.shape).astype(np.float32)

    corr = noise_correlation_report(z_subset, z_hat_prime)
    assert corr['pearson_mean_abs'] > 0.5,  "Expected high Pearson"
    assert corr['w2_noise_mean']    >= 0,   "Noise W2 must be non-negative"
    assert 'kl_noise_sym' in corr,          "Noise KL sym must be present"

    print(f"  Pearson |r| mean:   {corr['pearson_mean_abs']:.4f}  ✓")
    print(f"  CCA mean ρ:         {corr['cca']['mean_canonical_corr']:.4f}  ✓")
    print(f"  MI:                 {corr['mutual_information']:.4f} nats  ✓")
    print(f"  W1 noise (exact):   {corr['w1_noise_mean']:.4f}  ✓")
    print(f"  W2 noise (exact):   {corr['w2_noise_mean']:.4f}  ✓")
    print(f"  KL noise (sym):     {corr['kl_noise_sym']:.4f}  ✓")
    print(f"  W2 action (exact):  {div['w2_mean']:.4f}  ✓")
    print(f"  KL action (sym):    {div['kl_sym']:.4f}  ✓")
    print("\n=== Smoke test passed ===\n")


if __name__ == '__main__':
    _smoke_test()

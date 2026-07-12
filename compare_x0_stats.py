import argparse
from pathlib import Path

import numpy as np


def load_sampled_flat_samples(
    path: Path,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    arr = np.load(path, mmap_mode='r')
    if arr.ndim < 2:
        raise ValueError(f'Expected at least 2 dims in {path}, got shape {arr.shape}')

    flat = arr.reshape(-1, arr.shape[-1])
    total_rows = flat.shape[0]
    sample_size = min(total_rows, max_samples)

    rng = np.random.default_rng(seed)
    if sample_size < total_rows:
        indices = np.sort(rng.choice(total_rows, size=sample_size, replace=False))
        sample = np.asarray(flat[indices], dtype=np.float64)
    else:
        sample = np.asarray(flat, dtype=np.float64)

    return sample, arr.shape, total_rows


def summarize(name: str, x: np.ndarray) -> dict:
    norms = np.linalg.norm(x, axis=1)
    cov = np.cov(x, rowvar=False)
    diag = np.diag(cov)
    offdiag = cov - np.diag(diag)
    return {
        'name': name,
        'num_samples': x.shape[0],
        'dim': x.shape[1],
        'global_mean': float(x.mean()),
        'global_std': float(x.std()),
        'mean_abs_per_dim_mean': float(np.abs(x.mean(axis=0)).mean()),
        'mean_abs_per_dim_max': float(np.abs(x.mean(axis=0)).max()),
        'std_per_dim_mean': float(x.std(axis=0).mean()),
        'std_per_dim_min': float(x.std(axis=0).min()),
        'std_per_dim_max': float(x.std(axis=0).max()),
        'norm_mean': float(norms.mean()),
        'norm_std': float(norms.std()),
        'norm_q05': float(np.quantile(norms, 0.05)),
        'norm_q50': float(np.quantile(norms, 0.50)),
        'norm_q95': float(np.quantile(norms, 0.95)),
        'cov_trace': float(np.trace(cov)),
        'cov_diag_mean': float(diag.mean()),
        'cov_diag_std': float(diag.std()),
        'cov_offdiag_rms': float(np.sqrt(np.mean(offdiag**2))),
    }


def print_summary(stats: dict) -> None:
    print(f"== {stats['name']} ==")
    print(f"samples: {stats['num_samples']}")
    print(f"dim: {stats['dim']}")
    print(f"global_mean: {stats['global_mean']:.6f}")
    print(f"global_std: {stats['global_std']:.6f}")
    print(f"mean_abs_per_dim_mean: {stats['mean_abs_per_dim_mean']:.6f}")
    print(f"mean_abs_per_dim_max: {stats['mean_abs_per_dim_max']:.6f}")
    print(f"std_per_dim_mean: {stats['std_per_dim_mean']:.6f}")
    print(f"std_per_dim_min: {stats['std_per_dim_min']:.6f}")
    print(f"std_per_dim_max: {stats['std_per_dim_max']:.6f}")
    print(f"norm_mean: {stats['norm_mean']:.6f}")
    print(f"norm_std: {stats['norm_std']:.6f}")
    print(f"norm_q05: {stats['norm_q05']:.6f}")
    print(f"norm_q50: {stats['norm_q50']:.6f}")
    print(f"norm_q95: {stats['norm_q95']:.6f}")
    print(f"cov_trace: {stats['cov_trace']:.6f}")
    print(f"cov_diag_mean: {stats['cov_diag_mean']:.6f}")
    print(f"cov_diag_std: {stats['cov_diag_std']:.6f}")
    print(f"cov_offdiag_rms: {stats['cov_offdiag_rms']:.6f}")
    print()


def align_sample_counts(x0: np.ndarray, x0_hat: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x0), len(x0_hat))
    rng = np.random.default_rng(seed)

    x0_idx = rng.choice(len(x0), size=n, replace=False)
    x0_hat_idx = rng.choice(len(x0_hat), size=n, replace=False)

    return x0[x0_idx], x0_hat[x0_hat_idx]


def sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    num_projections: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(num_projections, x.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    dists = []
    for d in dirs:
        xp = np.sort(x @ d)
        yp = np.sort(y @ d)
        dists.append(np.mean(np.abs(xp - yp)))
    return float(np.mean(dists))


def compare(x0: np.ndarray, x0_hat: np.ndarray, num_projections: int, seed: int) -> None:
    x0_mean = x0.mean(axis=0)
    x0_hat_mean = x0_hat.mean(axis=0)
    x0_std = x0.std(axis=0)
    x0_hat_std = x0_hat.std(axis=0)

    mean_diff = np.abs(x0_mean - x0_hat_mean)
    std_diff = np.abs(x0_std - x0_hat_std)

    cov_x0 = np.cov(x0, rowvar=False)
    cov_x0_hat = np.cov(x0_hat, rowvar=False)
    cov_diff = cov_x0 - cov_x0_hat

    x0_norms = np.linalg.norm(x0, axis=1)
    x0_hat_norms = np.linalg.norm(x0_hat, axis=1)

    swd = sliced_wasserstein(x0, x0_hat, num_projections=num_projections, seed=seed)

    print('== comparison ==')
    print('note: this is distributional, not sample-by-sample, since x0_all and x0_hat_all are not row-aligned')
    print(f'mean_abs_diff_per_dim_mean: {mean_diff.mean():.6f}')
    print(f'mean_abs_diff_per_dim_max: {mean_diff.max():.6f}')
    print(f'std_abs_diff_per_dim_mean: {std_diff.mean():.6f}')
    print(f'std_abs_diff_per_dim_max: {std_diff.max():.6f}')
    print(f'norm_mean_diff: {abs(x0_norms.mean() - x0_hat_norms.mean()):.6f}')
    print(f'norm_std_diff: {abs(x0_norms.std() - x0_hat_norms.std()):.6f}')
    print(f"cov_frobenius_diff: {np.linalg.norm(cov_diff, ord='fro'):.6f}")
    print(f'cov_diag_mean_abs_diff: {np.abs(np.diag(cov_diff)).mean():.6f}')
    print(f'sliced_wasserstein: {swd:.6f}')
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare x0_all.npy and x0_hat_all.npy statistically.')
    parser.add_argument('x0_path', type=Path, help='Path to x0_all.npy')
    parser.add_argument('x0_hat_path', type=Path, help='Path to x0_hat_all.npy')
    parser.add_argument('--max-samples', type=int, default=50000, help='Max samples from each file')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for subsampling/projections')
    parser.add_argument('--num-projections', type=int, default=128, help='Sliced Wasserstein projections')
    args = parser.parse_args()

    if not args.x0_path.exists():
        raise FileNotFoundError(f'Missing file: {args.x0_path}')
    if not args.x0_hat_path.exists():
        raise FileNotFoundError(f'Missing file: {args.x0_hat_path}')

    x0, x0_shape, x0_total_rows = load_sampled_flat_samples(args.x0_path, max_samples=args.max_samples, seed=args.seed)
    x0_hat, x0_hat_shape, x0_hat_total_rows = load_sampled_flat_samples(
        args.x0_hat_path,
        max_samples=args.max_samples,
        seed=args.seed + 1,
    )

    print(f'loaded x0: {args.x0_path} -> original_shape={x0_shape}, flattened_rows={x0_total_rows}, sampled={x0.shape}')
    print(
        f'loaded x0_hat: {args.x0_hat_path} -> original_shape={x0_hat_shape}, '
        f'flattened_rows={x0_hat_total_rows}, sampled={x0_hat.shape}'
    )
    print()

    x0_cmp, x0_hat_cmp = align_sample_counts(x0, x0_hat, seed=args.seed)

    print(f'comparison sample count per file: {len(x0_cmp)}')
    print()

    print_summary(summarize('x0', x0_cmp))
    print_summary(summarize('x0_hat', x0_hat_cmp))
    compare(x0_cmp, x0_hat_cmp, num_projections=args.num_projections, seed=args.seed)


if __name__ == '__main__':
    main()

import argparse
from pathlib import Path

import numpy as np


def format_stats(name: str, array: np.ndarray) -> str:
    return (
        f'{name}: shape={array.shape}, dtype={array.dtype}, '
        f'min={array.min():.6f}, max={array.max():.6f}, '
        f'mean={array.mean():.6f}, std={array.std():.6f}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect a saved x1_all.npy file.')
    parser.add_argument('path', type=Path, help='Path to x1_all.npy')
    parser.add_argument(
        '--examples',
        type=int,
        default=3,
        help='Number of leading batches to summarize.',
    )
    parser.add_argument(
        '--tail-examples',
        type=int,
        default=1,
        help='Number of trailing batches to summarize.',
    )
    args = parser.parse_args()

    if not args.path.exists():
        raise FileNotFoundError(f'Could not find file: {args.path}')

    x1 = np.load(args.path, mmap_mode='r')
    print(f'path: {args.path.resolve()}')
    print(f'ndim: {x1.ndim}')
    print(f'shape: {x1.shape}')
    print(f'dtype: {x1.dtype}')
    print(f'file_size_bytes: {args.path.stat().st_size}')

    if x1.ndim < 2:
        raise ValueError(f'Expected x1 to have at least 2 dimensions, got shape {x1.shape}')

    flat = np.asarray(x1).reshape(x1.shape[0], -1)
    nonzero_rows = np.any(flat != 0, axis=1)
    num_nonzero_rows = int(nonzero_rows.sum())
    num_zero_rows = int((~nonzero_rows).sum())

    print(f'filled_steps: {num_nonzero_rows}')
    print(f'all_zero_steps: {num_zero_rows}')
    if num_nonzero_rows:
        last_filled_step = int(np.nonzero(nonzero_rows)[0][-1])
        print(f'last_filled_step_index: {last_filled_step}')
    else:
        print('last_filled_step_index: none')

    if num_nonzero_rows:
        filled = np.asarray(x1[nonzero_rows])
        print(format_stats('filled_global_stats', filled))
    else:
        print('filled_global_stats: no nonzero rows detected')

    head_count = min(args.examples, x1.shape[0])
    for idx in range(head_count):
        print(format_stats(f'head_step_{idx}', np.asarray(x1[idx])))

    tail_count = min(args.tail_examples, x1.shape[0])
    for offset in range(tail_count):
        idx = x1.shape[0] - tail_count + offset
        print(format_stats(f'tail_step_{idx}', np.asarray(x1[idx])))


if __name__ == '__main__':
    main()

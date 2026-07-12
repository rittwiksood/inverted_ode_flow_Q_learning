"""Demo runner for `run_subset_inversion_analysis`.
Creates a synthetic dataset and a DummyAgent implementing
`compute_reverse_flow_actions` so you can run the full pipeline
locally without a trained model.
"""

from subset_inversion_analysis import run_subset_inversion_analysis
import numpy as np
import os


class DummyAgent:
    def compute_reverse_flow_actions(self, obs_b, act_b):
        # Accepts jax arrays or numpy arrays; return numpy array of same shape
        a = np.array(act_b)
        M, d = a.shape
        return np.random.randn(M, d).astype(np.float32)


class SimpleDataset:
    def __init__(self, N=2000, s_dim=8, a_dim=4):
        self.size = N
        rng = np.random.RandomState(0)
        self.obs = rng.randn(N, s_dim).astype(np.float32)
        self.actions = rng.randn(N, a_dim).astype(np.float32)

    def get_subset(self, idx):
        return {'observations': self.obs[idx], 'actions': self.actions[idx]}


def main():
    out_dir = 'demo_outputs'
    os.makedirs(out_dir, exist_ok=True)

    dataset = SimpleDataset(N=2000)

    # Create a simple x0_all.npy (original training noises)
    x0_path = os.path.join(out_dir, 'x0_all.npy')
    if not os.path.exists(x0_path):
        np.save(x0_path, np.random.randn(dataset.size, dataset.actions.shape[1]).astype(np.float32))

    agent = DummyAgent()

    res = run_subset_inversion_analysis(
        agent=agent,
        dataset=dataset,
        x0_path=x0_path,
        strategy='random',
        removal_frac=0.3,
        max_samples=1000,
        save_dir=out_dir,
        seed=0,
    )

    print('\nDemo finished. Result keys:')
    print(list(res.keys()))
    np.save(os.path.join(out_dir, 'z_hat_prime_demo.npy'), res['z_hat_prime'])


if __name__ == '__main__':
    main()

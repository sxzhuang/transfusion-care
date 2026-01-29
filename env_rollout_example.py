from __future__ import annotations

import numpy as np

from env import TransfusionEnv
from utils import DynVAE_CatA


def main():
    model = DynVAE_CatA()
    dose_values = np.linspace(0.0, 1.0, model.K)
    env = TransfusionEnv(model, max_T=5, beta=0.1, dose_values=dose_values)

    rng = np.random.default_rng(0)
    z_t = model.sample_z1()
    env.reset(start_t=1, z_t=z_t)

    terminated = False
    step_idx = 0
    while not terminated:
        action = int(rng.integers(0, model.K))
        _, reward, terminated, _, info = env.step(action)
        print(
            f"step={step_idx} t={info['t']} a={action} "
            f"reward={reward:.4f} terminated={terminated}"
        )
        step_idx += 1


if __name__ == "__main__":
    main()

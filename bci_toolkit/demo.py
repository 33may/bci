from __future__ import annotations
import numpy as np

from .app import run_with_arrays

def demo():
    n_ch = 32
    sfreq = 250.0
    n_times = int(60 * sfreq)

    rng = np.random.default_rng(42)
    coords2d = rng.uniform(-0.08, 0.08, size=(n_ch, 2))

    t = np.arange(n_times) / sfreq
    data = 1e-6 * (0.6 * np.sin(2*np.pi*10*t)[None, :]
                   + 0.35 * np.sin(2*np.pi*22*t + rng.uniform(0, 2*np.pi, (n_ch, 1)))
                   + 0.07 * rng.standard_normal((n_ch, n_times)))

    ch_names = [f"Ch{idx:02d}" for idx in range(n_ch)]
    run_with_arrays(data, coords2d, sfreq, ch_names)

if __name__ == "__main__":
    demo()

"""Synthetic ECG generator - for TESTS and SMOKE TESTS only (never for results)."""
from __future__ import annotations

import numpy as np

# (centre offset from R [s], width [s], amplitude [mV])
_WAVES = [(-0.20, 0.025, 0.15), (-0.035, 0.010, -0.12), (0.0, 0.012, 1.0),
          (0.030, 0.012, -0.25), (0.25, 0.045, 0.30)]


def synth_ecg(duration_s: float, fs: float, rng: np.random.Generator,
              hr_bpm: float = 72.0, dc: float = 0.0):
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    r_idx = []
    tb = 0.5
    while tb < duration_s - 0.5:
        for off, w, a in _WAVES:
            x += a * np.exp(-0.5 * ((t - (tb + off)) / w) ** 2)
        r_idx.append(int(round(tb * fs)))
        tb += 60.0 / hr_bpm + rng.normal(0, 0.04)
    x += 0.005 * rng.standard_normal(n) + dc
    return x, np.array(r_idx)

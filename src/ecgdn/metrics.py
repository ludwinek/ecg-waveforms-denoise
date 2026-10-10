"""Signal-fidelity metrics.

Conventions (see paper, Section "Quality metrics"):

* All metrics are computed on the evaluation window only: ``warmup_s`` seconds are cut at BOTH
  ends (adaptive-filter convergence at the start; baseline-extrapolation and filter edge
  effects at both ends).
* ``remove_dc=True`` (default): the mean of the reference and of the processed
  signal is removed before comparison. The DC offset of a recording carries no
  diagnostic information, and band-pass filters remove it by construction -
  leaving it in would penalise every filter with a high-pass edge for an
  "error" that is not a distortion of the ECG morphology.
* SNR(ref, x) = 10 log10( P(ref) / P(ref - x) ).
* Delta-SNR = SNR(ref, filtered) - SNR(ref, noisy).
* PRD = 100 * ||ref - x|| / ||ref||  (percent root-mean-square difference).
"""
from __future__ import annotations

import numpy as np


def _prep(ref: np.ndarray, x: np.ndarray, remove_dc: bool):
    ref = np.asarray(ref, dtype=float)
    x = np.asarray(x, dtype=float)
    if ref.shape != x.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {x.shape}")
    if remove_dc:
        ref = ref - ref.mean()
        x = x - x.mean()
    return ref, x


def snr_db(ref: np.ndarray, x: np.ndarray, remove_dc: bool = True) -> float:
    ref, x = _prep(ref, x, remove_dc)
    p_sig = np.mean(ref**2)
    p_err = np.mean((ref - x) ** 2)
    return float(10.0 * np.log10(p_sig / max(p_err, 1e-30)))


def delta_snr_db(ref, noisy, filtered, remove_dc: bool = True) -> float:
    return snr_db(ref, filtered, remove_dc) - snr_db(ref, noisy, remove_dc)


def rmse(ref, x, remove_dc: bool = True) -> float:
    ref, x = _prep(ref, x, remove_dc)
    return float(np.sqrt(np.mean((ref - x) ** 2)))


def prd(ref, x, remove_dc: bool = True) -> float:
    ref, x = _prep(ref, x, remove_dc)
    return float(100.0 * np.linalg.norm(ref - x) / max(np.linalg.norm(ref), 1e-30))


def estimate_delay(ref: np.ndarray, x: np.ndarray, max_lag: int = 100) -> int:
    """Lag (samples) that maximises cross-correlation of x against ref.

    Positive value = x lags behind ref (typical for causal filters).
    """
    ref = np.asarray(ref, float) - np.mean(ref)
    x = np.asarray(x, float) - np.mean(x)
    best_lag, best_val = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            v = np.dot(ref[: len(ref) - lag], x[lag:])
        else:
            v = np.dot(ref[-lag:], x[: len(x) + lag])
        if v > best_val:
            best_lag, best_val = lag, v
    return best_lag


def compensate_delay(ref: np.ndarray, x: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Shift x to undo group delay (edge samples are padded with edge values)."""
    lag = estimate_delay(ref, x, max_lag)
    if lag == 0:
        return np.asarray(x, float)
    out = np.empty_like(x, dtype=float)
    if lag > 0:
        out[:-lag] = x[lag:]
        out[-lag:] = x[-1]
    else:
        out[-lag:] = x[:lag]
        out[:-lag] = x[0]
    return out


def evaluate(ref, noisy, filtered, fs: float, warmup_s: float = 10.0,
             remove_dc: bool = True) -> dict:
    """All fidelity metrics on the evaluation window (warm-up excluded)."""
    w = int(round(warmup_s * fs))
    r, n, f = ref[w:-w], noisy[w:-w], filtered[w:-w]
    return {
        "snr_in": snr_db(r, n, remove_dc),
        "snr_out": snr_db(r, f, remove_dc),
        "delta_snr": delta_snr_db(r, n, f, remove_dc),
        "rmse": rmse(r, f, remove_dc),
        "prd": prd(r, f, remove_dc),
    }

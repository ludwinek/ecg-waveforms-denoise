"""Classical (non-adaptive) denoisers: Butterworth, notch, median baseline, DWT."""
from __future__ import annotations

import numpy as np
import pywt
from scipy import ndimage, signal


def butterworth_bandpass(x, fs, low=0.5, high=40.0, order=4, zero_phase=True):
    """Butterworth band-pass; ``order`` is the prototype order N (the band-pass
    has 2N poles, as returned by scipy). zero_phase=False -> causal sosfilt."""
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    if zero_phase:
        return signal.sosfiltfilt(sos, x)
    # Causal: initialise the filter state to the steady state for the first sample,
    # otherwise a DC offset produces a long start-up transient that is not a
    # property of the filter's steady-state behaviour.
    x = np.asarray(x, float)
    y, _ = signal.sosfilt(sos, x, zi=signal.sosfilt_zi(sos) * x[0])
    return y


def notch_filter(x, fs, f0=50.0, q=30.0, zero_phase=True):
    b, a = signal.iirnotch(f0, q, fs=fs)
    sos = signal.tf2sos(b, a)
    if zero_phase:
        return signal.sosfiltfilt(sos, x)
    x = np.asarray(x, float)
    y, _ = signal.sosfilt(sos, x, zi=signal.sosfilt_zi(sos) * x[0])
    return y


def _odd(n: int) -> int:
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def median_baseline(x, fs, w1_s=0.2, w2_s=0.6):
    """Classic cascade of two median filters (200 ms, 600 ms) estimates the
    baseline; the output is the signal minus that estimate."""
    b = ndimage.median_filter(x, size=_odd(w1_s * fs), mode="nearest")
    b = ndimage.median_filter(b, size=_odd(w2_s * fs), mode="nearest")
    return x - b


def dwt_denoise(x, fs=None, wavelet="db6", level=8, zero_approx=True,
                zero_details=(), thr_levels=(1, 2, 3, 4, 5), mode="soft", thr_scale=1.0):
    """DWT denoising.

    * Approximation at ``level`` is zeroed if ``zero_approx`` (baseline removal).
      At fs=360 Hz the level-L approximation covers 0 .. fs/2^(L+1) Hz:
      L=8 -> ~0.70 Hz, L=9 -> ~0.35 Hz (does NOT contain a 0.5 Hz drift).
    * Detail levels in ``zero_details`` are removed outright (needed for
      narrow-band, high-amplitude interference: thresholding keeps large
      coefficients, so it cannot remove a strong 50 Hz sinusoid).
    * Detail levels in ``thr_levels`` are thresholded with the universal
      threshold (scaled by ``thr_scale``), sigma estimated by MAD from the finest scale.
    """
    x = np.asarray(x, float)
    n = len(x)
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="symmetric")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = thr_scale * sigma * np.sqrt(2.0 * np.log(n))
    for lev in range(1, level + 1):
        idx = level - lev + 1  # coeffs = [cA_L, cD_L, ..., cD_1]
        if lev in zero_details:
            coeffs[idx] = np.zeros_like(coeffs[idx])
        elif lev in thr_levels:
            coeffs[idx] = pywt.threshold(coeffs[idx], thr, mode=mode)
    if zero_approx:
        coeffs[0] = np.zeros_like(coeffs[0])
    return pywt.waverec(coeffs, wavelet, mode="symmetric")[:n]

"""Uniform registry: every method is f(noisy, fs, ctx, **params) -> filtered.

ctx keys: noise_type, oracle_ref (noisy copy of the true noise; ORACLE methods
only - an upper bound, not realisable in practice).
Parameters in DEFAULTS are the legacy/starting values; the paper uses values
tuned on the DEV records only (see scripts/tune_dev.py, to be added).
"""
from __future__ import annotations

import numpy as np

from . import adaptive, classical

ALL_NOISE = ("bw", "em", "ma", "pli")


def _ac(x):
    """Static DC removal before adaptation (equivalent to AC coupling).

    A DC offset in the primary input cannot be modelled by a zero-mean noise
    reference, so NLMS/RLS keep trying to cancel it and the weights drift
    (observed: Delta-SNR of about -15 dB at 5 dB input SNR for an ECG with a
    1 mV offset, vs about +7 dB without it). Applied to ALL adaptive methods.
    """
    x = np.asarray(x, float)
    return x - x.mean()


def _none(noisy, fs, ctx, **p):
    return np.asarray(noisy, float)


def _butter(noisy, fs, ctx, zero_phase=True, **p):
    return classical.butterworth_bandpass(noisy, fs, zero_phase=zero_phase, **p)


def _notch(noisy, fs, ctx, **p):
    return classical.notch_filter(noisy, fs, **p)


def _median(noisy, fs, ctx, **p):
    return classical.median_baseline(noisy, fs, **p)


def _dwt(noisy, fs, ctx, **p):
    return classical.dwt_denoise(noisy, fs, **p)


def _lms_oracle(noisy, fs, ctx, taps=16, mu=0.005, **p):
    return adaptive.nlms_anc(_ac(noisy), adaptive.delay_line(_ac(ctx["oracle_ref"]), taps), mu=mu, **p)


def _rls_oracle(noisy, fs, ctx, taps=16, lam=0.995, delta=1.0, **p):
    return adaptive.rls_anc(_ac(noisy), adaptive.delay_line(_ac(ctx["oracle_ref"]), taps), lam=lam, delta=delta)


def _lms_blind(noisy, fs, ctx, f0=50.0, harmonics=(1,), mu=0.01, **p):
    U = adaptive.quadrature(len(noisy), fs, f0, harmonics)
    return adaptive.nlms_anc(_ac(noisy), U, mu=mu, **p)


def _rls_blind(noisy, fs, ctx, f0=50.0, harmonics=(1,), lam=0.995, delta=1.0, **p):
    U = adaptive.quadrature(len(noisy), fs, f0, harmonics)
    return adaptive.rls_anc(_ac(noisy), U, lam=lam, delta=delta)


# name -> (function, noise types it applies to, default params)
REGISTRY = {
    "none":           (_none,       ALL_NOISE, {}),
    "butter_zp":      (_butter,     ALL_NOISE, dict(low=0.5, high=40.0, order=4, zero_phase=True)),
    "butter_causal":  (_butter,     ALL_NOISE, dict(low=0.5, high=40.0, order=4, zero_phase=False)),
    "notch":          (_notch,      ("pli",),  dict(f0=50.0, q=30.0)),
    "median_bl":      (_median,     ("bw",),   dict(w1_s=0.2, w2_s=0.6)),
    "dwt":            (_dwt,        ALL_NOISE, dict(wavelet="db6", level=8, zero_approx=True,
                                                   zero_details=(), thr_levels=(1, 2, 3, 4, 5))),
    "dwt_legacy":     (_dwt,        ALL_NOISE, dict(wavelet="db6", level=5, zero_approx=False,
                                                   zero_details=(), thr_levels=(1, 2, 3, 4, 5))),
    "lms_oracle":     (_lms_oracle, ALL_NOISE, dict(taps=16, mu=0.005)),
    "rls_oracle":     (_rls_oracle, ALL_NOISE, dict(taps=16, lam=0.995, delta=1.0)),
    "lms_blind":      (_lms_blind,  ("pli",),  dict(f0=50.0, harmonics=(1,), mu=0.01)),
    "rls_blind":      (_rls_blind,  ("pli",),  dict(f0=50.0, harmonics=(1,), lam=0.995, delta=1.0)),
}

CAUSAL = {"butter_causal"}  # need group-delay compensation before scoring


def run(name: str, noisy, fs, ctx, overrides: dict | None = None):
    fn, _, defaults = REGISTRY[name]
    params = {**defaults, **(overrides or {})}
    return fn(noisy, fs, ctx, **params)


def applicable(name: str, noise_type: str) -> bool:
    return noise_type in REGISTRY[name][1]

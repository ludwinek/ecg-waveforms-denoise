"""Adaptive noise cancellers (NLMS, RLS), numba-accelerated.

Primary input d = noisy ECG, regressors U (n x m). Output = error e = d - w.u,
i.e. the ECG estimate. Two regressor builders:

* delay line of a (possibly noisy) noise reference  -> "oracle" / reference ANC
* quadrature sin/cos at the NOMINAL line frequency   -> "blind" adaptive notch
"""
from __future__ import annotations

import numpy as np
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view


def delay_line(ref: np.ndarray, taps: int) -> np.ndarray:
    padded = np.concatenate([np.zeros(taps - 1), np.asarray(ref, float)])
    return np.ascontiguousarray(sliding_window_view(padded, taps))


def quadrature(n: int, fs: float, f0: float = 50.0, harmonics=(1,)) -> np.ndarray:
    t = np.arange(n) / fs
    cols = []
    for h in harmonics:
        cols += [np.sin(2 * np.pi * h * f0 * t), np.cos(2 * np.pi * h * f0 * t)]
    return np.ascontiguousarray(np.stack(cols, axis=1))


@njit(cache=True)
def _nlms(d, U, mu, eps):
    n, m = U.shape
    w = np.zeros(m)
    e = np.empty(n)
    for i in range(n):
        y = 0.0
        nrm = eps
        for k in range(m):
            y += w[k] * U[i, k]
            nrm += U[i, k] * U[i, k]
        err = d[i] - y
        e[i] = err
        g = mu * err / nrm
        for k in range(m):
            w[k] += g * U[i, k]
    return e


@njit(cache=True)
def _rls(d, U, lam, delta):
    n, m = U.shape
    w = np.zeros(m)
    P = np.eye(m) / delta
    e = np.empty(n)
    for i in range(n):
        u = U[i]
        Pu = P @ u
        k = Pu / (lam + u @ Pu)
        err = d[i] - w @ u
        w = w + k * err
        P = (P - np.outer(k, Pu)) / lam
        P = 0.5 * (P + P.T)  # keep P symmetric (numerical stability)
        e[i] = err
    return e


def nlms_anc(d, U, mu=0.005, eps=1e-8):
    return _nlms(np.asarray(d, float), np.ascontiguousarray(U, dtype=float), float(mu), float(eps))


def rls_anc(d, U, lam=0.995, delta=1.0):
    return _rls(np.asarray(d, float), np.ascontiguousarray(U, dtype=float), float(lam), float(delta))

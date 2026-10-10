"""Noise generation and SNR control.

Noise sources
-------------
* ``bw``, ``em``, ``ma``: REAL noise from the MIT-BIH Noise Stress Test
  Database (NSTDB). Use :func:`load_nstdb_noise` + :func:`nstdb_segment`.
* ``pli``: extended synthetic power-line interference (fundamental + 3rd
  harmonic, slow frequency drift, amplitude modulation): :func:`synth_pli`.
* ``synth_*_like`` generators are stand-ins used ONLY for unit tests and
  smoke tests when the NSTDB files are not available. Never report results
  obtained with them.
"""
from __future__ import annotations

import os

import numpy as np
from scipy import signal


# ----------------------------------------------------------------------------
# SNR control
# ----------------------------------------------------------------------------
def power(x: np.ndarray, remove_dc: bool = True) -> float:
    x = np.asarray(x, float)
    if remove_dc:
        x = x - x.mean()
    return float(np.mean(x**2))


def scale_to_snr(ref: np.ndarray, v: np.ndarray, snr_db: float,
                 remove_dc: bool = True) -> np.ndarray:
    """Scale noise ``v`` (and remove its mean) so that SNR(ref, ref+v) = snr_db.

    Signal power is computed on the mean-removed reference, consistently with
    :mod:`ecgdn.metrics`.
    """
    v = np.asarray(v, float) - np.mean(v)
    p_sig = power(ref, remove_dc)
    p_v = power(v, remove_dc=False)
    if p_v <= 0:
        raise ValueError("noise has zero power")
    gain = np.sqrt(p_sig / (p_v * 10.0 ** (snr_db / 10.0)))
    return v * gain


def measured_snr_db(ref: np.ndarray, noisy: np.ndarray, remove_dc: bool = True) -> float:
    ref = np.asarray(ref, float)
    err = np.asarray(noisy, float) - ref
    return 10.0 * np.log10(power(ref, remove_dc) / power(err, remove_dc))


def add_noise(ref: np.ndarray, v: np.ndarray, snr_db: float, tol_db: float = 1e-6):
    """Return (noisy, scaled_noise) and *verify* the realised SNR."""
    vs = scale_to_snr(ref, v, snr_db)
    noisy = ref + vs
    got = measured_snr_db(ref, noisy)
    if abs(got - snr_db) > tol_db:
        raise AssertionError(f"SNR_in verification failed: target {snr_db}, got {got}")
    return noisy, vs


def oracle_reference(v: np.ndarray, snr_ref_db: float, rng: np.random.Generator) -> np.ndarray:
    """Noisy copy of the true noise: ref = v + white, with P(v)/P(white)=snr_ref."""
    w = rng.standard_normal(len(v))
    w *= np.sqrt(power(v, False) / (power(w, False) * 10.0 ** (snr_ref_db / 10.0)))
    return v + w


# ----------------------------------------------------------------------------
# Extended synthetic power-line interference
# ----------------------------------------------------------------------------
def _slow_noise(n: int, fs: float, cutoff_hz: float, rng: np.random.Generator) -> np.ndarray:
    sos = signal.butter(2, cutoff_hz, btype="low", fs=fs, output="sos")
    x = signal.sosfilt(sos, rng.standard_normal(n + int(10 * fs / cutoff_hz)))
    x = x[-n:]
    return x / (np.max(np.abs(x)) + 1e-12)


def synth_pli(n: int, fs: float, rng: np.random.Generator, f0: float = 50.0,
              harmonic_amp: float = 0.2, freq_drift_hz: float = 0.2,
              am_depth: float = 0.2) -> np.ndarray:
    """50 Hz + 3rd harmonic, slowly drifting frequency, amplitude modulation."""
    dev = freq_drift_hz * _slow_noise(n, fs, 0.05, rng)
    phase = 2 * np.pi * np.cumsum(f0 + dev) / fs + rng.uniform(0, 2 * np.pi)
    t = np.arange(n) / fs
    f_am = rng.uniform(0.1, 0.5)
    am = 1.0 + am_depth * np.sin(2 * np.pi * f_am * t + rng.uniform(0, 2 * np.pi))
    v = am * (np.sin(phase) + harmonic_amp * np.sin(3 * phase + rng.uniform(0, 2 * np.pi)))
    return v - v.mean()


def legacy_sine(n: int, fs: float, f: float, rng: np.random.Generator) -> np.ndarray:
    """Idealised pure sinusoid (previous version of the study; repo-only)."""
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))


# ----------------------------------------------------------------------------
# Stand-ins for NSTDB noise (tests / smoke tests ONLY)
# ----------------------------------------------------------------------------
def synth_bw_like(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / fs
    resp = np.sin(2 * np.pi * rng.uniform(0.15, 0.35) * t + rng.uniform(0, 6.28))
    return _slow_noise(n, fs, 0.3, rng) + 0.7 * resp


def synth_em_like(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    base = _slow_noise(n, fs, 8.0, rng) * 0.3
    # sparse transients with exponential recovery (electrode motion)
    spikes = np.zeros(n)
    k = max(1, int(n / fs / 15))
    spikes[rng.integers(0, n, k)] = rng.normal(0, 3.0, k)
    sos = signal.butter(1, 1.0, btype="low", fs=fs, output="sos")
    return base + signal.sosfilt(sos, spikes) * 5.0


def synth_ma_like(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    sos = signal.butter(4, [10, 100], btype="band", fs=fs, output="sos")
    x = signal.sosfilt(sos, rng.standard_normal(n))
    env = 0.5 + np.abs(_slow_noise(n, fs, 0.2, rng))
    return x * env


SYNTH_STANDINS = {"bw": synth_bw_like, "em": synth_em_like, "ma": synth_ma_like}


# ----------------------------------------------------------------------------
# NSTDB (real noise)
# ----------------------------------------------------------------------------
def load_nstdb_noise(data_dir: str, kind: str, channel: int = 0):
    """Load an NSTDB noise record ('bw', 'em' or 'ma'); returns (signal_mV, fs)."""
    import wfdb

    if kind not in {"bw", "em", "ma"}:
        raise ValueError(kind)
    rec = wfdb.rdrecord(os.path.join(data_dir, kind))
    x = rec.p_signal[:, channel].astype(float)
    return x - x.mean(), float(rec.fs)


def nstdb_segment(noise: np.ndarray, fs: float, n: int, window_s: tuple,
                  rng: np.random.Generator) -> np.ndarray:
    """Random length-``n`` segment taken strictly inside ``window_s`` (seconds).

    The noise records are split in TIME (train window / test window) so a DL
    model never sees the evaluation noise during training.
    """
    a, b = int(window_s[0] * fs), int(window_s[1] * fs)
    if b - a < n:
        raise ValueError("noise window shorter than requested segment")
    start = int(rng.integers(a, b - n + 1))
    return noise[start:start + n].copy()

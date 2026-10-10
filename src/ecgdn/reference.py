"""Construction of the 'reference ECG' from a raw MIT-BIH segment.

Raw MIT-BIH records are NOT clean: they contain baseline drift (often 25-55 % of the
AC power of a 5-min segment lies below 0.5 Hz). Scoring against the raw record
(i) caps the output SNR of any filter with a high-pass edge at ~2.5-6 dB and
(ii) rewards methods that leave the drift in place.  The reference is therefore
drift-corrected once per record, before noise is added.

Modes
-----
spline    (main)  natural cubic-spline baseline through isoelectric knots in the PR
                  segment of every annotated beat (Meyer & Keiser 1977 style), subtracted.
                  Independent of every filter compared in the study.
          Limitation: knots arrive once per beat (~1.2 Hz), so drift components above ~0.3 Hz
          are only partly captured (accuracy degrades towards the knot Nyquist rate).
spline_hp spline correction followed by a zero-phase 2nd-order Butterworth high-pass at
          ``spline_hp_cutoff_hz`` (default 0.3 Hz, i.e. BELOW the 0.5 Hz edge of the compared
          filters): removes the sub-band residue the knots cannot capture.
highpass  (sensitivity analysis)  zero-phase 2nd-order Butterworth high-pass at
          ``highpass_cutoff_hz`` (default 0.5 Hz).
raw       (reporting only)  DC removed, nothing else.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ")     # all beat annotations (used for QRS evaluation)
KNOT_SYMBOLS = set("NLRBAaJSejn")           # beats with a reliable isoelectric PR segment


def beat_positions(ann_samples, ann_symbols, n, symbols=BEAT_SYMBOLS):
    s = np.asarray(ann_samples, dtype=int)
    keep = np.array([c in symbols for c in ann_symbols], dtype=bool)
    s = s[keep]
    return s[(s >= 0) & (s < n)]


def spline_baseline(x, fs, beats, pr_offset_s=0.08, win_s=0.02, flat_factor=3.0):
    """Baseline through isoelectric knots (mean of a ``win_s`` window centred
    ``pr_offset_s`` before each beat). Windows that are not flat (std > flat_factor x the
    median window std: P-wave onset, noise burst, ectopy) are rejected. No smoothing of the
    knot values - drift (e.g. respiration, 0.2-0.3 Hz) is only ~3-4 knots per cycle."""
    x = np.asarray(x, float)
    n = len(x)
    half = max(1, int(round(win_s * fs / 2)))
    kt, ky, ks = [], [], []
    for r in beats:
        c = int(round(r - pr_offset_s * fs))
        if c - half < 0 or c + half + 1 > n:
            continue
        w = x[c - half:c + half + 1]
        kt.append(c); ky.append(float(np.mean(w))); ks.append(float(np.std(w)))
    if len(kt) < 4:
        raise ValueError("too few beats to build a baseline")
    kt, ky, ks = np.array(kt, float), np.array(ky), np.array(ks)
    good = ks <= flat_factor * max(np.median(ks), 1e-9)
    if good.sum() >= 4:
        kt, ky = kt[good], ky[good]
    o = np.argsort(kt)
    kt, ky = kt[o], ky[o]
    u = np.concatenate([[True], np.diff(kt) > 0])
    kt, ky = kt[u], ky[u]
    f = CubicSpline(kt, ky, bc_type="natural")   # PCHIP clips between-knot extrema of drift
    return f(np.clip(np.arange(n), kt[0], kt[-1]))


def make_reference(x, fs, ann_samples=None, ann_symbols=None, mode="spline", pr_offset_s=0.08,
                   spline_hp_cutoff_hz=0.3, highpass_cutoff_hz=0.5):
    """Return (reference, info). The reference has zero mean."""
    x = np.asarray(x, float)
    if mode == "raw":
        ref = x.copy()
    elif mode == "highpass":
        sos = signal.butter(2, highpass_cutoff_hz, btype="highpass", fs=fs, output="sos")
        ref = signal.sosfiltfilt(sos, x)
    elif mode == "spline":
        beats = beat_positions(ann_samples, ann_symbols, len(x), KNOT_SYMBOLS)
        ref = x - spline_baseline(x, fs, beats, pr_offset_s)
    elif mode == "spline_hp":
        beats = beat_positions(ann_samples, ann_symbols, len(x), KNOT_SYMBOLS)
        ref = x - spline_baseline(x, fs, beats, pr_offset_s)
        sos = signal.butter(2, spline_hp_cutoff_hz, btype="highpass", fs=fs, output="sos")
        ref = signal.sosfiltfilt(sos, ref)
    else:
        raise ValueError(mode)
    ref = ref - ref.mean()
    p_raw = np.var(x)
    info = {"mode": mode, "power_removed_pct": 100.0 * (1.0 - np.var(ref) / p_raw) if p_raw > 0 else 0.0}
    return ref, info

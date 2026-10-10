"""QRS detection and scoring against the MIT-BIH beat annotations.

Detector: wfdb.processing.xqrs_detect (public, deterministic, standard in the community).
Scoring: one-to-one matching within a tolerance (default 150 ms, as in AAMI EC57),
Se / +P / F1, median R-peak localisation error, and the relative error of RMSSD computed from
the detected vs the annotated beat series (false/missed detections corrupt HRV).
Only beats inside the evaluation window [lo, hi) are scored.
"""
from __future__ import annotations

import numpy as np
from wfdb import processing


def detect_qrs(sig, fs):
    try:
        d = processing.xqrs_detect(sig=np.asarray(sig, float), fs=float(fs), verbose=False)
    except Exception:
        d = []
    return np.asarray(d, dtype=int)


def match_beats(ref, det, tol):
    """Greedy one-to-one matching of sorted sample indices within +-tol samples."""
    ref, det = np.sort(ref), np.sort(det)
    i = j = 0
    pairs = []
    while i < len(ref) and j < len(det):
        d = det[j] - ref[i]
        if abs(d) <= tol:
            pairs.append((i, j)); i += 1; j += 1
        elif d < -tol:
            j += 1
        else:
            i += 1
    return pairs


def _rmssd(samples, fs):
    rr = np.diff(np.sort(samples)) / fs * 1000.0
    return float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 2 else float("nan")


def qrs_scores(ref_beats, det_beats, fs, lo, hi, tol_s=0.15):
    ref = np.asarray(ref_beats, int); det = np.asarray(det_beats, int)
    ref = ref[(ref >= lo) & (ref < hi)]
    det = det[(det >= lo) & (det < hi)]
    pairs = match_beats(ref, det, int(round(tol_s * fs)))
    tp = len(pairs); fn = len(ref) - tp; fp = len(det) - tp
    loc = (float(np.median([abs(det[j] - ref[i]) for i, j in pairs])) / fs * 1000.0) if pairs else float("nan")
    r_ref, r_det = _rmssd(ref, fs), _rmssd(det, fs)
    return {"tp": tp, "fp": fp, "fn": fn,
            "se": tp / (tp + fn) if tp + fn else float("nan"),
            "ppv": tp / (tp + fp) if tp + fp else float("nan"),
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan"),
            "loc_err_ms": loc,
            "rmssd_err_pct": abs(r_det - r_ref) / r_ref * 100.0 if r_ref and np.isfinite(r_ref) and np.isfinite(r_det) else float("nan")}

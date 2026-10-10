"""MIT-BIH loading (requires the files from PhysioNet; see scripts/download_data.py)."""
from __future__ import annotations

import os

import numpy as np


def load_mitbih_segment(data_dir: str, record: str, channel_name: str = "MLII",
                        seconds: float = 300.0, start_s: float = 0.0):
    """Return (signal_mV, fs, annotation_samples_in_segment, annotation_symbols); channel selected BY NAME."""
    import wfdb

    path = os.path.join(data_dir, record)
    hdr = wfdb.rdheader(path)
    if channel_name not in hdr.sig_name:
        raise KeyError(f"{record}: channel {channel_name} not in {hdr.sig_name}")
    ch = hdr.sig_name.index(channel_name)
    fs = float(hdr.fs)
    a, b = int(start_s * fs), int((start_s + seconds) * fs)
    rec = wfdb.rdrecord(path, sampfrom=a, sampto=b, channels=[ch])
    ann = wfdb.rdann(path, "atr", sampfrom=a, sampto=b)
    return rec.p_signal[:, 0].astype(float), fs, np.asarray(ann.sample) - a, list(ann.symbol)

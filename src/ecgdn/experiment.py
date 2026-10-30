"""One experimental condition = (record, noise type, SNR_in): add noise with verified
SNR, run every applicable method on the identical noisy signal, score."""
from __future__ import annotations

import numpy as np

from . import methods
from .metrics import compensate_delay, evaluate, prd, snr_db
from .noise import add_noise, nstdb_segment, oracle_reference, synth_pli

NO_ORACLE_NEEDED = [m for m in methods.REGISTRY if "oracle" not in m]


def run_condition(ref_ecg, noise_raw, noise_type, snr_in, fs, method_names, rng,
                  snr_ref_db=10.0, warmup_s=10.0, remove_dc=True, overrides=None):
    noisy, v = add_noise(ref_ecg, noise_raw, snr_in)
    ctx = {"noise_type": noise_type, "oracle_ref": oracle_reference(v, snr_ref_db, rng)}
    overrides = overrides or {}
    rows = []
    for name in method_names:
        if not methods.applicable(name, noise_type):
            continue
        out = methods.run(name, noisy, fs, ctx, overrides.get(name))
        if name in methods.CAUSAL:
            out = compensate_delay(ref_ecg, out)
        rows.append({"method": name, "noise": noise_type, "snr_target": snr_in,
                     **evaluate(ref_ecg, noisy, out, fs, warmup_s, remove_dc)})
    return rows


def filter_only_distortion(ref_ecg, fs, method_names=None, warmup_s=10.0,
                           remove_dc=True, overrides=None):
    """Run each non-oracle method on the reference ECG WITHOUT added noise.

    Whatever error remains is distortion caused by the filter itself (plus any
    genuine noise that is already inside the 'reference'). This is the floor
    that makes Delta-SNR negative at high input SNR.
    """
    overrides = overrides or {}
    rows = []
    for name in method_names or NO_ORACLE_NEEDED:
        out = methods.run(name, ref_ecg, fs, {"noise_type": None, "oracle_ref": None},
                          overrides.get(name))
        if name in methods.CAUSAL:
            out = compensate_delay(ref_ecg, out)
        w = int(round(warmup_s * fs))
        rows.append({"method": name,
                     "prd_noise_free": prd(ref_ecg[w:-w], out[w:-w], remove_dc),
                     "snr_noise_free": snr_db(ref_ecg[w:-w], out[w:-w], remove_dc)})
    return rows


NT_REAL = ("bw", "em", "ma")


def make_noisy(ref, noise_type, snr, rec, ni, si, fs, cfg, nstdb):
    """Single source of truth for the evaluation noisy signal of (record, noise type, SNR).

    Used by run_test.py (classical/adaptive methods) AND run_dl.py, so every method sees
    bit-identical input. Returns (noisy, scaled_noise, rng); the rng state is the one the
    oracle reference is drawn from afterwards.
    """
    rng = np.random.default_rng([cfg["seed"], int(rec), ni, si, 1])
    n = len(ref)
    if noise_type == "pli":
        raw = synth_pli(n, fs, rng, **{k: cfg["pli"][k] for k in ("f0", "harmonic_amp", "freq_drift_hz", "am_depth")})
    else:
        raw = nstdb_segment(nstdb[noise_type], fs, n, cfg["nstdb"]["test_seconds"], rng)
    noisy, v = add_noise(ref, raw, snr)
    return noisy, v, rng

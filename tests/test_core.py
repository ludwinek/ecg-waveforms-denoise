import numpy as np
import pytest

from ecgdn import methods, metrics, noise, splits, synth
from ecgdn.experiment import filter_only_distortion, run_condition

FS = 360.0


@pytest.fixture(scope="module")
def ecg():
    rng = np.random.default_rng(1)
    x, r = synth.synth_ecg(120, FS, rng, dc=1.0)
    return x


@pytest.mark.parametrize("snr", [0, 5, 10, 20])
def test_snr_control_is_exact(ecg, snr):
    rng = np.random.default_rng(2)
    v = noise.synth_pli(len(ecg), FS, rng)
    noisy, _ = noise.add_noise(ecg, v, snr)
    assert abs(metrics.snr_db(ecg, noisy) - snr) < 1e-6


def test_identity_has_zero_delta_snr(ecg):
    rng = np.random.default_rng(3)
    noisy, _ = noise.add_noise(ecg, noise.synth_bw_like(len(ecg), FS, rng), 10)
    assert abs(metrics.delta_snr_db(ecg, noisy, noisy)) < 1e-9


def test_dc_handling(ecg):
    """With DC removal a pure DC change is not an error; without it, it is."""
    shifted = ecg - ecg.mean()
    assert metrics.rmse(ecg, shifted, remove_dc=True) < 1e-12
    assert metrics.rmse(ecg, shifted, remove_dc=False) > 0.5


def test_delay_compensation(ecg):
    shifted = np.concatenate([np.zeros(7), ecg[:-7]])
    fixed = metrics.compensate_delay(ecg, shifted)
    assert metrics.prd(ecg[100:-100], fixed[100:-100]) < 1.0


def test_pli_spectrum_has_fundamental_and_harmonic():
    rng = np.random.default_rng(4)
    v = noise.synth_pli(int(FS * 60), FS, rng)
    f = np.fft.rfftfreq(len(v), 1 / FS)
    P = np.abs(np.fft.rfft(v)) ** 2
    band = lambda lo, hi: P[(f > lo) & (f < hi)].sum()
    assert band(49, 51) > 10 * band(100, 130)
    assert band(148, 152) > 0.01 * band(49, 51)


def test_adaptive_notch_beats_butterworth_on_pli(ecg):
    rng = np.random.default_rng(5)
    v = noise.synth_pli(len(ecg), FS, rng)
    rows = run_condition(ecg, v, "pli", 5, FS, ["butter_zp", "notch", "lms_blind", "lms_oracle"], rng)
    d = {r["method"]: r["delta_snr"] for r in rows}
    assert d["lms_oracle"] > 8 and d["notch"] > 3 and d["lms_blind"] > 3


def test_rls_oracle_runs_and_improves_bw(ecg):
    rng = np.random.default_rng(6)
    v = noise.synth_bw_like(len(ecg), FS, rng)
    rows = run_condition(ecg, v, "bw", 5, FS, ["rls_oracle", "median_bl", "butter_zp"], rng)
    d = {r["method"]: r["delta_snr"] for r in rows}
    assert d["rls_oracle"] > 3 and d["median_bl"] > 3 and d["butter_zp"] > 3


def test_dwt_level_covers_baseline_band(ecg):
    """Legacy level-5 DWT cannot remove a 0.5 Hz drift; level-8 approx zeroing can."""
    rng = np.random.default_rng(7)
    v = noise.legacy_sine(len(ecg), FS, 0.5, rng)
    rows = run_condition(ecg, v, "bw", 10, FS, ["dwt_legacy", "dwt"], rng)
    d = {r["method"]: r["delta_snr"] for r in rows}
    assert d["dwt_legacy"] < 1.0
    assert d["dwt"] > 5.0


def test_filter_only_distortion_small_for_butterworth(ecg):
    rows = {r["method"]: r for r in filter_only_distortion(ecg, FS)}
    assert rows["butter_zp"]["prd_noise_free"] < 15


def test_splits_deterministic_and_disjoint():
    recs = [str(i) for i in range(48)]
    a = splits.make_splits(recs, ["0", "1", "2", "3"], 8, 4, 123)
    b = splits.make_splits(recs, ["0", "1", "2", "3"], 8, 4, 123)
    assert a == b
    assert len(a["dev"]) == 8 and len(a["test"]) == 36
    assert not set(a["dev"]) & set(a["test"])
    assert sorted(sum(a["folds"], [])) == a["test"]
    assert all(len(f) == 9 for f in a["folds"])


def test_nstdb_segment_stays_in_window():
    rng = np.random.default_rng(8)
    x = np.arange(360 * 1800, dtype=float)
    seg = noise.nstdb_segment(x, 360.0, 360 * 300, (1200, 1800), rng)
    assert seg[0] >= 1200 * 360 and seg[-1] < 1800 * 360


@pytest.mark.parametrize("name", ["lms_oracle", "rls_oracle", "lms_blind", "rls_blind"])
def test_adaptive_methods_invariant_to_dc_offset(name):
    rng = np.random.default_rng(9)
    x0, _ = synth.synth_ecg(60, FS, rng, dc=0.0)
    v = noise.synth_pli(len(x0), FS, np.random.default_rng(10))
    out = {}
    for dc in (0.0, 1.0):
        noisy, vs = noise.add_noise(x0 + dc, v, 5)
        ctx = {"oracle_ref": noise.oracle_reference(vs, 10, np.random.default_rng(11))}
        y = methods.run(name, noisy, FS, ctx)
        out[dc] = metrics.delta_snr_db(x0 + dc, noisy, y)
    assert abs(out[0.0] - out[1.0]) < 0.1


def test_spline_reference_recovers_clean_ecg_under_strong_drift():
    from ecgdn.reference import make_reference
    rng = np.random.default_rng(12)
    clean, r = synth.synth_ecg(120, FS, rng, dc=0.0)
    tt = np.arange(len(clean)) / FS
    drift = 0.6 * np.sin(2 * np.pi * 0.12 * tt) + 0.25 * np.sin(2 * np.pi * 0.22 * tt + 1.0) \
        + 0.5 * noise._slow_noise(len(clean), FS, 0.05, rng)
    raw = clean + drift + 0.8
    ref, info = make_reference(raw, FS, r, ["N"] * len(r), "spline")
    c0 = clean - clean.mean()
    assert abs(ref.mean()) < 1e-9
    w = int(10 * FS)                                 # evaluation window: 10 s trimmed at both ends
    assert metrics.prd(c0[w:-w], ref[w:-w]) < 5.0    # spline recovers the ECG (drift < ~0.25 Hz)
    assert metrics.prd(c0[w:-w], (raw - raw.mean())[w:-w]) > 80.0  # raw would be badly off
    assert info["power_removed_pct"] > 30


def test_highpass_reference_runs():
    from ecgdn.reference import make_reference
    rng = np.random.default_rng(13)
    x, r = synth.synth_ecg(60, FS, rng, dc=1.0)
    ref, _ = make_reference(x, FS, r, ["N"] * len(r), "highpass")
    assert abs(ref.mean()) < 1e-9


def test_match_beats_one_to_one_and_scores():
    from ecgdn.qrs import match_beats, qrs_scores
    ref = np.array([100, 400, 700, 1000])
    det = np.array([102, 398, 405, 1500])        # one duplicate near 400, one false at 1500, one missed (700, 1000)
    pairs = match_beats(ref, det, 10)
    assert pairs == [(0, 0), (1, 1)]
    s = qrs_scores(ref, det, 360.0, 0, 2000, tol_s=10 / 360)
    assert (s["tp"], s["fp"], s["fn"]) == (2, 2, 2)
    assert abs(s["f1"] - 0.5) < 1e-9


def test_xqrs_finds_beats_on_clean_synthetic_ecg():
    from ecgdn.qrs import detect_qrs, qrs_scores
    rng = np.random.default_rng(14)
    x, r = synth.synth_ecg(120, FS, rng)
    s = qrs_scores(r, detect_qrs(x, FS), FS, int(10 * FS), len(x) - int(10 * FS))
    assert s["f1"] > 0.97 and s["loc_err_ms"] < 30


def test_dwt_threshold_scale_monotone_shrinkage(ecg):
    from ecgdn import classical
    rng = np.random.default_rng(15)
    noisy, _ = noise.add_noise(ecg, noise.synth_ma_like(len(ecg), FS, rng), 5)
    base = dict(wavelet="db6", level=8, thr_levels=(1, 2, 3, 4, 5), mode="soft")
    e = [np.sum(np.abs(classical.dwt_denoise(noisy, **base, thr_scale=s))) for s in (0.3, 0.6, 1.0)]
    assert e[0] >= e[1] >= e[2]


# ------------------------------------------------------------------ DL data side (numpy only)
def test_dl_overlap_add_identity_reconstructs_signal():
    from ecgdn.dl_data import denoise_signal
    rng = np.random.default_rng(16)
    x, _ = synth.synth_ecg(30, FS, rng, dc=1.0)
    y = denoise_signal(lambda a: a, x)               # identity "model"
    assert metrics.prd(x - x.mean(), y) < 0.01       # identity model -> exact reconstruction


def test_dl_sampler_realises_requested_snr_and_normalisation():
    from ecgdn.dl_data import NoiseBank, TrainSampler
    rng = np.random.default_rng(17)
    ref, _ = synth.synth_ecg(120, FS, rng)
    bank = NoiseBank({k: noise.SYNTH_STANDINS[k](int(FS * 300), FS, rng) for k in ("bw", "em", "ma")},
                     np.stack([noise.synth_pli(int(FS * 100), FS, rng) for _ in range(3)]))
    sm = TrainSampler([ref], bank, snr_range=(10.0, 10.0))
    X, Y = sm.batch(64, np.random.default_rng(1))
    assert X.shape == (64, 1, 720) and X.dtype == np.float32
    assert np.allclose(X.std(axis=2), 1.0, atol=1e-3)            # normalised by noisy std
    err = X - Y                                                   # = noise / sd
    snr = 10 * np.log10((Y ** 2).mean(axis=2) / (err ** 2).mean(axis=2))   # power incl. window offset
    assert np.allclose(snr, 10.0, atol=1e-3)


def test_make_noisy_is_deterministic_and_verified():
    from ecgdn.experiment import make_noisy
    import yaml, pathlib
    cfg = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1] / "config" / "protocol.yaml"))
    rng = np.random.default_rng(18)
    ref, _ = synth.synth_ecg(60, FS, rng)
    a, v1, _ = make_noisy(ref, "pli", 5, "100", 3, 1, FS, cfg, {})
    b, v2, _ = make_noisy(ref, "pli", 5, "100", 3, 1, FS, cfg, {})
    assert np.array_equal(a, b) and abs(metrics.snr_db(ref, a) - 5) < 1e-6

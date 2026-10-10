"""Smoke test on SYNTHETIC data (no downloads needed). Verifies the pipeline end to end.
NOT a result: synthetic ECG and stand-in noise only."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, yaml
from ecgdn import methods, noise, synth
from ecgdn.experiment import filter_only_distortion, run_condition
from ecgdn.metrics import evaluate

cfg = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1] / "config" / "protocol.yaml"))
fs, seed = cfg["fs"], cfg["seed"]
names = list(methods.REGISTRY)
rows, dist = [], []
for rec in range(3):
    rng = np.random.default_rng(seed + rec)
    ecg, _ = synth.synth_ecg(120, fs, rng, hr_bpm=60 + 12 * rec, dc=1.0)  # 1 mV offset on purpose
    n = len(ecg)
    for nt in cfg["noise_types"]:
        for snr in cfg["snr_in_db"]:
            raw = (noise.synth_pli(n, fs, rng, **{k: cfg["pli"][k] for k in ("f0", "harmonic_amp", "freq_drift_hz", "am_depth")})
                   if nt == "pli" else noise.SYNTH_STANDINS[nt](n, fs, rng))
            for r in run_condition(ecg, raw, nt, snr, fs, names, rng,
                                   snr_ref_db=cfg["oracle_reference"]["snr_ref_db"],
                                   warmup_s=cfg["warmup_seconds"]):
                rows.append({"record": rec, **r})
    for d in filter_only_distortion(ecg, fs, warmup_s=cfg["warmup_seconds"]):
        dist.append({"record": rec, **d})

df = pd.DataFrame(rows)
df.to_csv("results/smoke_synthetic.csv", index=False)
pd.set_option("display.width", 200)
for nt in cfg["noise_types"]:
    t = df[df.noise == nt].pivot_table(index="method", columns="snr_target", values="delta_snr", aggfunc="mean").round(2)
    print(f"\n=== mean delta-SNR (dB), noise={nt}  [synthetic stand-in]\n{t}")
print("\n=== filter-only distortion (PRD % on noise-free reference)")
print(pd.DataFrame(dist).groupby("method")["prd_noise_free"].mean().round(2))

# Metric pitfall demo: Butterworth at 20 dB with / without DC removal in the metric
rng = np.random.default_rng(seed)
ecg, _ = synth.synth_ecg(120, fs, rng, dc=1.0)
noisy, _ = noise.add_noise(ecg, noise.synth_pli(len(ecg), fs, rng), 20)
out = methods.run("butter_zp", noisy, fs, {})
for rd in (True, False):
    print(f"Butterworth, PLI 20 dB, remove_dc={rd}: delta-SNR = {evaluate(ecg, noisy, out, fs, remove_dc=rd)['delta_snr']:.2f} dB")

"""Diagnostics to run BEFORE freezing tuned parameters.  Usage: python scripts/diagnose.py data

A) per-SNR delta-SNR of the best configuration found by tune_dev (results/tuning_dev.csv)
B) filter-only distortion on REAL dev records (no added noise): PRD of each filter alone,
   over the whole evaluation window and with 30 s trimmed at both ends (edge effects).
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, yaml
from scipy import signal
from ecgdn import data, methods
from ecgdn.metrics import compensate_delay, prd

pd.set_option("display.width", 220, "display.max_rows", 200, "display.max_colwidth", 90)
cfg = yaml.safe_load(open(ROOT / "config" / "protocol.yaml"))
data_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data")

# ---- A
csv = ROOT / "results" / "tuning_dev.csv"
if csv.exists():
    df = pd.read_csv(csv)
    best = (df.groupby(["method", "noise", "params"])["delta_snr"].mean().reset_index()
              .sort_values("delta_snr", ascending=False).groupby(["method", "noise"]).head(1))
    rows = []
    for _, b in best.iterrows():
        sub = df[(df.method == b.method) & (df.noise == b.noise) & (df.params == b.params)]
        per = sub.groupby("snr")["delta_snr"].mean().round(2)
        rows.append({"method": b.method, "noise": b.noise, **{f"{int(k)} dB": v for k, v in per.items()},
                     "mean": round(b.delta_snr, 2)})
    print("A) best config per method/noise - mean delta-SNR by input SNR\n",
          pd.DataFrame(rows).sort_values(["method", "noise"]).to_string(index=False))
else:
    print("A) results/tuning_dev.csv not found - skipped")

# ---- B
from ecgdn.reference import make_reference
sp = yaml.safe_load(open(ROOT / "config" / "splits.yaml"))
fs = cfg["fs"]; w = int(cfg["warmup_seconds"] * fs); t30 = int(30 * fs)
configs = [
    ("butter_zp", "butter_zp", {}), ("butter_causal", "butter_causal", {}), ("notch", "notch", {}),
    ("median_bl", "median_bl", {}),
    ("dwt L8 soft thr1-5 (default)", "dwt", {}),
    ("dwt L8 no thresholding", "dwt", dict(thr_levels=())),
    ("dwt L9 no thresholding", "dwt", dict(level=9, thr_levels=())),
    ("dwt L8 hard thr1-5", "dwt", dict(mode="hard")),
    ("dwt_legacy (L5, approx kept)", "dwt_legacy", {}),
    ("lms_blind (1)", "lms_blind", {}), ("rls_blind (1)", "rls_blind", {}),
]
raws = {}
for rec in sp["dev"]:
    raws[rec] = data.load_mitbih_segment(str(data_dir / "mitdb"), rec, cfg["channel_name"], cfg["segment_seconds"])

R = cfg["reference"]
modes = ["raw", "spline", "spline_hp", "highpass"]
tab = {c[0]: {} for c in configs}
perrec = []
for mode in modes:
    cols = {c[0]: [] for c in configs}
    for rec, (x, _, ann_s, ann_sym) in raws.items():
        ref, info = make_reference(x, fs, ann_s, ann_sym, mode, R["pr_offset_s"],
                                   R["spline_hp_cutoff_hz"], R["highpass_cutoff_hz"])
        sos = signal.butter(2, 0.5, btype="highpass", fs=fs, output="sos")
        lf = 100.0 * (1 - np.var(signal.sosfiltfilt(sos, ref)[w:-w]) / np.var(ref[w:-w]))
        row = {"record": rec, "mode": mode, "drift_removed_%": round(info["power_removed_pct"], 1),
               "ref_power_below_0.5Hz_%": round(lf, 1)}
        for label, name, ov in configs:
            out = methods.run(name, ref, fs, {"noise_type": None, "oracle_ref": None}, ov)
            if name in methods.CAUSAL:
                out = compensate_delay(ref, out)
            p_ = prd(ref[w:-w], out[w:-w])
            cols[label].append(p_)
            if label == "butter_zp":
                row["PRD_butter_zp_%"] = round(p_, 1)
        perrec.append(row)
    for label in cols:
        tab[label][mode] = round(float(np.mean(cols[label])), 2)

t = pd.DataFrame(tab).T[modes]
t.columns = ["PRD% raw", "PRD% spline", f"PRD% spline_hp({R['spline_hp_cutoff_hz']}Hz)", f"PRD% highpass({R['highpass_cutoff_hz']}Hz)"]
print("\nB) filter-only distortion on dev ECG (noise-free), mean PRD over dev records, by reference mode\n", t.to_string())
pr = pd.DataFrame(perrec)
for mode in ("spline", "spline_hp"):
    print(f"\nC) per-record, reference mode = {mode}\n",
          pr[pr["mode"] == mode].drop(columns="mode").to_string(index=False))
print("\nReading guide: distortion alone caps the output SNR at about -20*log10(PRD/100) dB "
      "(PRD 10% -> 20 dB, 30% -> 10.5 dB).")

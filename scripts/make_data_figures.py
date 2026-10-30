"""Figures that need the raw data (and the trained U-Net for the DL trace).

Usage: python scripts/make_data_figures.py data [--record 100 --noise em --snr 5] [--no-dl]
Outputs (results/figures/): fig5_reference, fig6_example  (PDF + PNG), and results/reference_stats.csv
  * reference_stats.csv (raw_lf_pct, ref_lf_pct, removed_pct): for all 44 usable records, share of AC power below 0.5 Hz in the RAW segment
    and power removed by the drift-corrected reference (numbers for the paper text / abstract).
  * fig5_reference: raw segment with its spline baseline, the drift-corrected reference, and the
    distribution of drift power over the 44 records.
  * fig6_example: one segment: reference, noisy, and selected methods (incl. U-Net unless --no-dl).
"""
import argparse, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, yaml
from scipy import signal
from ecgdn import data, methods, noise
from ecgdn.dl_data import denoise_signal
from ecgdn.experiment import NT_REAL, make_noisy
from ecgdn.metrics import evaluate
from ecgdn.noise import oracle_reference
from ecgdn.reference import KNOT_SYMBOLS, beat_positions, make_reference, spline_baseline

W = 12.2 / 2.54
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "axes.titlesize": 7.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
ap = argparse.ArgumentParser()
ap.add_argument("data_dir", nargs="?", default="data")
ap.add_argument("--record", default="100")
ap.add_argument("--noise", default="em")
ap.add_argument("--snr", type=int, default=5)
ap.add_argument("--start", type=float, default=60.0, help="start of the plotted window (s)")
ap.add_argument("--length", type=float, default=8.0)
ap.add_argument("--seconds", type=float, default=None, help="segment length override (testing only)")
ap.add_argument("--no-dl", action="store_true")
a = ap.parse_args()

cfg = yaml.safe_load(open(ROOT / "config" / "protocol.yaml"))
sp = yaml.safe_load(open(ROOT / "config" / "splits.yaml"))
tuned = yaml.safe_load(open(ROOT / "config" / "tuned_params.yaml")) or {}
fs, R = cfg["fs"], cfg["reference"]
mitdb = str(pathlib.Path(a.data_dir) / "mitdb")
seg = a.seconds or cfg["segment_seconds"]
w = int(cfg["warmup_seconds"] * fs)
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load(rec):
    raw, _, ann_s, ann_sym = data.load_mitbih_segment(mitdb, rec, cfg["channel_name"], seg)
    ref, info = make_reference(raw, fs, ann_s, ann_sym, R["mode"], R["pr_offset_s"],
                               R["spline_hp_cutoff_hz"], R["highpass_cutoff_hz"])
    return raw, ref, info, ann_s, ann_sym


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ reference statistics
hp = signal.butter(2, 0.5, btype="highpass", fs=fs, output="sos")
rows = []
cache = {}
for rec in [r for r in cfg["records_all"] if r not in cfg["excluded_records"]]:
    raw, ref, info, ann_s, ann_sym = load(rec)
    cache[rec] = (raw, ann_s, ann_sym)
    x = raw[w:-w]
    lf = 100 * (1 - np.var(signal.sosfiltfilt(hp, raw)[w:-w]) / np.var(x))
    lf_ref = 100 * (1 - np.var(signal.sosfiltfilt(hp, ref)[w:-w]) / np.var(ref[w:-w]))
    rows.append({"record": rec, "set": "dev" if rec in sp["dev"] else "test", "raw_lf_pct": lf,
                 "ref_lf_pct": lf_ref, "removed_pct": info["power_removed_pct"]})
st = pd.DataFrame(rows)
st.to_csv(ROOT / "results" / "reference_stats.csv", index=False)
print(f"\nShare of AC power below 0.5 Hz in RAW 5-min segments ({len(st)} records):")
print(st.groupby("set").raw_lf_pct.agg(["mean", "median", "max"]).round(1).to_string())
print(f"all records: mean {st.raw_lf_pct.mean():.1f} %, median {st.raw_lf_pct.median():.1f} %, "
      f"max {st.raw_lf_pct.max():.1f} % ({st.loc[st.raw_lf_pct.idxmax(), 'record']}); "
      f"records above 50 %: {(st.raw_lf_pct > 50).sum()}; "
      f"after correction (reference): mean {st.ref_lf_pct.mean():.1f} %")

# ------------------------------------------------------------------ fig 5: reference construction
rec5 = st.loc[st.raw_lf_pct.idxmax(), "record"]
raw, ann_s, ann_sym = cache[rec5]
ref, _ = make_reference(raw, fs, ann_s, ann_sym, R["mode"], R["pr_offset_s"], R["spline_hp_cutoff_hz"], R["highpass_cutoff_hz"])
beats = beat_positions(ann_s, ann_sym, len(raw), KNOT_SYMBOLS)
base = spline_baseline(raw, fs, beats, R["pr_offset_s"])
i0 = int(max(w, 60 * fs)); i1 = i0 + int(30 * fs)
t = np.arange(i0, i1) / fs
fig = plt.figure(figsize=(W, 3.6))
gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], hspace=0.55, wspace=0.32)
a1 = fig.add_subplot(gs[0, 0])
a2 = fig.add_subplot(gs[1, 0], sharex=a1)
a3 = fig.add_subplot(gs[:, 1])
a1.plot(t, raw[i0:i1] - raw.mean(), color="0.35", lw=0.5, label="raw record")
a1.plot(t, base[i0:i1] - raw.mean(), color="#D55E00", lw=1.2, label="spline baseline")
a1.set_ylim(top=(raw[i0:i1] - raw.mean()).max() + 0.55)
a1.set_title(f"Record {rec5}: raw signal and baseline", pad=3)
a1.legend(frameon=False, ncol=2, loc="upper left", fontsize=5.8, handlelength=1.3, columnspacing=1.0, borderaxespad=0.1)
a1.set_ylabel("mV")
plt.setp(a1.get_xticklabels(), visible=False)
a2.plot(t, ref[i0:i1], color="#0072B2", lw=0.5)
a2.set_title("Drift-corrected reference ECG", pad=3)
a2.set_xlabel("Time (s)"); a2.set_ylabel("mV")
a3.hist(st.raw_lf_pct, bins=np.arange(0, 70, 5), color="0.55", edgecolor="white", lw=0.4)
a3.axvline(st.raw_lf_pct.mean(), color="#D55E00", lw=1.2, label=f"mean {st.raw_lf_pct.mean():.1f} %")
a3.legend(frameon=False, fontsize=6)
a3.set_xlabel("AC power below 0.5 Hz (%)"); a3.set_ylabel("Number of records")
a3.set_title(f"Raw 5-min segments, n = {len(st)}", pad=3)
save(fig, "fig5_reference")

# ------------------------------------------------------------------ fig 6: example
rec, nt, snr = a.record, a.noise, a.snr
ni, si = cfg["noise_types"].index(nt), cfg["snr_in_db"].index(snr)
raw, ref, _, ann_s, ann_sym = load(rec)
nst = {k: noise.load_nstdb_noise(str(pathlib.Path(a.data_dir) / "nstdb"), k)[0] for k in NT_REAL}
noisy, v, rng = make_noisy(ref, nt, snr, rec, ni, si, fs, cfg, nst)
ctx = {"noise_type": nt, "oracle_ref": oracle_reference(v, cfg["oracle_reference"]["snr_ref_db"], rng)}
names = ["butter_zp", "dwt", "lms_oracle" if nt == "pli" else "rls_oracle"]
labels = {"butter_zp": "Butterworth 0.5-40 Hz", "dwt": "DWT (tuned)", "rls_oracle": "RLS, oracle ref.",
          "lms_oracle": "LMS, oracle ref.", "dl_unet": "U-Net (blind)"}
outs = {m: methods.run(m, noisy, fs, ctx, (tuned.get(m) or {}).get(nt)) for m in names}
if not a.no_dl:
    import torch
    from ecgdn import dl_model
    k = next(i for i, f in enumerate(sp["folds"]) if rec in f)
    net = dl_model.UNet1D()
    net.load_state_dict(torch.load(ROOT / "models" / f"dl_fold{k}.pt", map_location="cpu"))
    outs["dl_unet"] = denoise_signal(dl_model.make_predictor(net, torch.device("cpu")), noisy)
j0 = int(a.start * fs); j1 = j0 + int(a.length * fs)
tt = np.arange(j0, j1) / fs
panels = [("Reference ECG", ref, None), (f"Noisy ({nt.upper()}, {snr} dB)", noisy, None)] + \
         [(labels[m], o, evaluate(ref, noisy, o, fs, cfg["warmup_seconds"])["delta_snr"]) for m, o in outs.items()]
lim = (ref[j0:j1].min() - 0.4, ref[j0:j1].max() + 0.4)
fig, axs = plt.subplots(len(panels), 1, figsize=(W, 0.95 * len(panels)), sharex=True)
for ax, (title, sig, d_snr) in zip(axs, panels):
    sig = sig - sig.mean() if title.startswith("Noisy") else sig
    ax.plot(tt, ref[j0:j1], color="0.75", lw=0.6)
    ax.plot(tt, (sig - np.mean(sig[w:-w]))[j0:j1] if title != "Reference ECG" else ref[j0:j1],
            color="#000000" if title.startswith(("Ref", "Noisy")) else "#0072B2", lw=0.6)
    ax.set_ylim(*lim)
    ax.set_title(title + (f"  ($\\Delta$SNR = {d_snr:.1f} dB over the whole record)" if d_snr is not None else ""),
                 fontsize=6.5, loc="left", pad=1.5)
    ax.set_ylabel("mV", fontsize=6)
axs[-1].set_xlabel("Time (s)")
fig.tight_layout()
fig.subplots_adjust(bottom=0.1)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color="#000000", lw=0.8), Line2D([], [], color="#0072B2", lw=0.8), Line2D([], [], color="0.75", lw=1.2)],
           ["reference ECG / noisy input", "method output", "reference ECG (grey, for comparison)"],
           loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=3, frameon=False, fontsize=6)
save(fig, "fig6_example")

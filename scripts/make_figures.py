"""Publication figures from results/*.csv (no raw data needed).

Usage: python scripts/make_figures.py
Writes PDF (vector) + PNG (300 dpi) to results/figures/. Width 12.2 cm = Springer LNNS text width.
  fig1_dsnr       mean delta-SNR vs input SNR, one panel per noise type, 95 % bootstrap CI over records
  fig2_f1_gain    change in QRS-detection F1 vs no filtering at 0 dB input SNR (95 % bootstrap CI)
  fig3_proxy      delta-SNR gain vs F1 gain, every record x condition x method (SNR is a poor proxy)
  fig4_floor      distortion of each filter on the noise-free reference (sets the delta-SNR ceiling)
"""
import pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np, pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)
W = 12.2 / 2.54   # inches
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "lines.linewidth": 1.0,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
RNG = np.random.default_rng(20261008)

t = pd.read_csv(RES / "test_results.csv", dtype={"record": str})
d = t[t.method != "reference"].copy()
if (RES / "dl_results.csv").exists():
    d = pd.concat([d, pd.read_csv(RES / "dl_results.csv", dtype={"record": str})], ignore_index=True)

NOISE = {"bw": "Baseline wander (NSTDB)", "em": "Electrode motion (NSTDB)", "ma": "Muscle artifact (NSTDB)",
         "pli": "Power-line interference (synthetic)"}
# method: (label, colour, linestyle, marker)   - Okabe-Ito colour-blind-safe palette
STY = {"butter_zp": ("Butterworth 0.5-40 Hz", "#0072B2", "-", "o"),
       "butter_causal": ("Butterworth (causal)", "#56B4E9", ":", "o"),
       "median_bl": ("Median baseline", "#56B4E9", "-", "s"),
       "notch": ("Notch 50 Hz", "#009E73", "-", "^"),
       "dwt": ("DWT (tuned)", "#CC79A7", "-", "D"),
       "dwt_legacy": ("DWT level 5 (legacy)", "#CC79A7", ":", "D"),
       "lms_blind": ("LMS blind", "#E69F00", "-", "v"),
       "rls_blind": ("RLS blind", "#8C8C8C", "-", "x"),
       "lms_oracle": ("LMS, oracle ref.", "#D55E00", "--", "o"),
       "rls_oracle": ("RLS, oracle ref.", "#D55E00", "-.", "^"),
       "dl_unet": ("U-Net (blind)", "#000000", "-", "*")}
SHOW = {"bw": ["butter_zp", "median_bl", "dwt", "lms_oracle", "rls_oracle", "dl_unet"],
        "em": ["butter_zp", "dwt", "lms_oracle", "rls_oracle", "dl_unet"],
        "ma": ["butter_zp", "dwt", "lms_oracle", "rls_oracle", "dl_unet"],
        "pli": ["butter_zp", "notch", "dwt", "lms_blind", "rls_blind", "lms_oracle", "rls_oracle", "dl_unet"]}
SNRS = [0, 5, 10, 20]
SHORT = {"butter_zp": "Butter.", "median_bl": "Median", "notch": "Notch", "dwt": "DWT", "dwt_legacy": "DWT-leg.",
         "lms_blind": "LMS-blind", "rls_blind": "RLS-blind", "lms_oracle": "LMS-oracle", "rls_oracle": "RLS-oracle",
         "dl_unet": "U-Net"}


def proxy(m):
    lab, col, ls, mk = STY[m]
    return Line2D([], [], color=col, ls=ls, marker=mk, ms=3.5 if mk != "*" else 6, lw=1.8 if m == "dl_unet" else 1.0)


GROUPS = [("Blind, classical", ["butter_zp", "median_bl", "notch", "dwt", "dwt_legacy"], 0.02),
          ("Blind, adaptive", ["lms_blind", "rls_blind"], 0.34),
          ("Adaptive, oracle ref.", ["lms_oracle", "rls_oracle"], 0.56),
          ("Blind, deep learning", ["dl_unet"], 0.80)]


def grouped_legend(fig, present, y=0.0):
    """One small legend per method family, side by side below the axes."""
    for title, ms, x in GROUPS:
        ms = [m for m in ms if m in present]
        if not ms:
            continue
        leg = fig.legend([proxy(m) for m in ms], [STY[m][0] for m in ms], title=title, loc="upper left",
                         bbox_to_anchor=(x, y), frameon=False, fontsize=5.8, title_fontsize=6.2, borderaxespad=0)
        leg._legend_box.align = "left"
        fig.add_artist(leg)


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    b = x[RNG.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ fig 1
fig, axs = plt.subplots(2, 2, figsize=(W, 4.6), sharex=True)
used = {}
for ax, nt in zip(axs.ravel(), NOISE):
    for m in SHOW[nt]:
        if m not in d.method.unique():
            continue
        lab, col, ls, mk = STY[m]
        g = d[(d.noise == nt) & (d.method == m)]
        mean, lo, hi = [], [], []
        for s in SNRS:
            v = g[g.snr_target == s].delta_snr.values
            mean.append(v.mean()); l, h = boot_ci(v); lo.append(l); hi.append(h)
        mean, lo, hi = map(np.array, (mean, lo, hi))
        lw = 1.8 if m == "dl_unet" else 1.0
        h = ax.errorbar(SNRS, mean, yerr=[mean - lo, hi - mean], color=col, ls=ls, marker=mk, ms=3.5 if mk != "*" else 6,
                        lw=lw, capsize=1.2, elinewidth=0.5, zorder=5 if m == "dl_unet" else 3)
        used[m] = h
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_title(NOISE[nt])
    ax.set_xticks(SNRS)
    ax.grid(alpha=0.25, lw=0.4)
for ax in axs[1]:
    ax.set_xlabel("Input SNR (dB)")
for ax in axs[:, 0]:
    ax.set_ylabel(r"$\Delta$SNR (dB)")
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
grouped_legend(fig, set(used), y=0.085)
fig.text(0.5, -0.115, "Markers: mean over the 36 test records; error bars: 95% bootstrap CI over records. "
         "Dashed / dash-dotted lines: oracle noise reference.", ha="center", fontsize=5.8)
save(fig, "fig1_dsnr")

# ------------------------------------------------------------------ fig 2
none = d[d.method == "none"].set_index(["record", "noise", "snr_target"]).f1
fig, axs = plt.subplots(1, 4, figsize=(W, 2.9), sharey=True)
used2 = {}
for ax, nt in zip(axs, NOISE):
    ms = [m for m in STY if m != "none" and ((d.noise == nt) & (d.method == m)).any() and m not in ("butter_causal",)]
    for i, m in enumerate(ms):
        g = d[(d.noise == nt) & (d.method == m) & (d.snr_target == 0)].set_index(["record", "noise", "snr_target"]).f1
        diff = (g - none.reindex(g.index)).values
        lo, hi = boot_ci(diff)
        col = STY[m][1]
        ax.errorbar(i, diff.mean(), yerr=[[diff.mean() - lo], [hi - diff.mean()]], color=col, marker=STY[m][3],
                    ms=3.5 if STY[m][3] != "*" else 6, capsize=1.5, elinewidth=0.7)
        used2[m] = 1
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([SHORT[m] for m in ms], rotation=90, ha="center", fontsize=5.5)
    ax.set_title(nt.upper() if nt != "pli" else "PLI")
    ax.grid(axis="y", alpha=0.25, lw=0.4)
axs[0].set_ylabel(r"$\Delta$F1 vs no filtering (0 dB input SNR)")
fig.tight_layout()
fig.subplots_adjust(bottom=0.33)
grouped_legend(fig, set(used2), y=0.02)
fig.text(0.5, -0.27, "Dots: mean F1 difference to the unfiltered noisy signal over the 36 test records; whiskers: 95% bootstrap CI "
         "(not corrected for multiple testing). Zero = no benefit.", ha="center", fontsize=5.6)
save(fig, "fig2_f1_gain")

# ------------------------------------------------------------------ fig 3
g = d[d.method != "none"].merge(d[d.method == "none"][["record", "noise", "snr_target", "f1"]],
                                on=["record", "noise", "snr_target"], suffixes=("", "_none"))
g["dF1"] = g.f1 - g.f1_none
fig, axs = plt.subplots(2, 2, figsize=(W, 4.6), sharex=True, sharey=True)
for ax, nt in zip(axs.ravel(), NOISE):
    q = g[g.noise == nt]
    ax.scatter(q.delta_snr, q.dF1, s=2.5, alpha=0.18, color="#0072B2", lw=0, rasterized=True)
    bins = np.arange(-15, 31, 3)
    idx = np.digitize(q.delta_snr, bins)
    bx = [q.delta_snr[idx == i].mean() for i in range(1, len(bins)) if (idx == i).sum() >= 15]
    by = [q.dF1[idx == i].mean() for i in range(1, len(bins)) if (idx == i).sum() >= 15]
    ax.plot(bx, by, color="#D55E00", marker="o", ms=2.5, lw=1.0)
    rho, p = stats.spearmanr(q.delta_snr, q.dF1)
    ax.text(0.03, 0.05, rf"Spearman $\rho$ = {rho:.2f}", transform=ax.transAxes, fontsize=6.5)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_title(NOISE[nt])
    ax.grid(alpha=0.25, lw=0.4)
for ax in axs[1]:
    ax.set_xlabel(r"$\Delta$SNR (dB)")
for ax in axs[:, 0]:
    ax.set_ylabel(r"$\Delta$F1 vs no filtering")
fig.tight_layout()
fig.subplots_adjust(bottom=0.14)
fig.legend([Line2D([], [], marker="o", ls="", color="#0072B2", ms=3, alpha=0.5),
            Line2D([], [], marker="o", ls="-", color="#D55E00", ms=3)],
           ["one method x record x input SNR (all methods, all SNRs)", "mean in 3 dB bins of $\\Delta$SNR (bins with >= 15 points)"],
           loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=2, frameon=False, fontsize=6)
save(fig, "fig3_proxy")

# ------------------------------------------------------------------ fig 4
if (RES / "test_distortion.csv").exists():
    x = pd.read_csv(RES / "test_distortion.csv", dtype={"record": str})
    x["label"] = x.method.map(lambda m: STY.get(m, (m,))[0])
    x.loc[(x.method == "dwt") & (x.noise == "bw"), "label"] = "DWT (tuned for BW)"
    x.loc[(x.method == "dwt") & (x.noise == "pli"), "label"] = "DWT (tuned for PLI)"
    x.loc[(x.method == "dwt") & (x.noise == "em"), "label"] = "DWT (tuned for EM)"
    x.loc[(x.method == "dwt") & (x.noise == "ma"), "label"] = "DWT (tuned for MA)"
    x = x[x.method != "none"].drop_duplicates(["record", "label"])
    order = x.groupby("label").prd_noise_free.median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.boxplot([x[x.label == l].prd_noise_free for l in order], vert=False, tick_labels=order, showfliers=False,
               boxprops=dict(lw=0.6), medianprops=dict(color="#D55E00", lw=1.0), whiskerprops=dict(lw=0.6),
               capprops=dict(lw=0.6))
    ax.set_xscale("log")
    for prd, lab in ((10 ** (-20 / 20) * 100, "20 dB"), (10 ** (-10 / 20) * 100, "10 dB")):
        ax.axvline(prd, color="0.5", ls=":", lw=0.7)
        ax.text(prd, len(order) + 0.55, f"SNR cap {lab}", ha="center", fontsize=5.5, color="0.35")
    ax.set_xlabel("PRD on noise-free reference ECG (%), log scale")
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    ax.legend([Patch(facecolor="white", edgecolor="black", lw=0.6), Line2D([], [], color="#D55E00", lw=1.0),
               Line2D([], [], color="0.5", ls=":", lw=0.7)],
              ["box: interquartile range\n(whiskers: min-max)", "median over records",
               "output-SNR cap\nat this distortion"], loc="center right", bbox_to_anchor=(1.0, 0.33), frameon=False, fontsize=5.8)
    fig.tight_layout()
    save(fig, "fig4_floor")

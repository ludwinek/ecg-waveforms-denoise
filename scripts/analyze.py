"""Statistical analysis of results/test_results.csv (unit of analysis = RECORD, n=36).

Usage: python scripts/analyze.py [results/test_results.csv] [results/test_distortion.csv] [--exclude 201] [--out analysis_excl201]
Writes tables to results/<out>/ (default results/analysis/) and prints the main ones.
--exclude drops the listed test records (e.g. 201, the same subject as development record 202) from every analysis.

Comparison families, each corrected with Holm within the family. P1 and P4 were planned in the protocol before testing
(not externally registered); P2 and P3 are SECONDARY: their comparator (blind LMS; Butterworth) was fixed after the classical
test results had been inspected. Everything else is exploratory.
  P1  LMS vs RLS, both oracle, delta-SNR            (4 noise x 4 SNR = 16 tests)
  P2  best blind (lms_blind) vs Butterworth zero-phase on PLI, delta-SNR   (4 tests)
  P3  DL (dl_unet) vs Butterworth zero-phase, delta-SNR   (4 noise x 4 SNR = 16 tests) - only if
      results/dl_results.csv exists. Butterworth is the universal blind baseline, fixed BEFORE seeing DL results.
  P4  every method vs no filtering, QRS F1          (noise x SNR x method)
Exploratory (labelled as such, Holm within each family):
  E1  oracle premium on PLI: oracle vs blind counterparts, and blind vs notch
  E5  DL vs the other blind methods (dwt; notch/lms_blind/rls_blind on PLI; median_bl on BW) and
      the gap to the best oracle method (rls_oracle for bw/em/ma, lms_oracle for PLI)
  E2  Friedman test per condition on delta-SNR across methods
  E3  Spearman correlation between delta-SNR gain and F1 gain over no filtering
  E4  RMSSD error (HRV) of each method vs no filtering; the noise-free detector already has a
      non-zero RMSSD error (missed/extra beats), so "excess" = error - error on the reference ECG
"""
import pathlib, sys
import numpy as np, pandas as pd
from scipy import stats

import argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser()
_ap.add_argument("results", nargs="?")
_ap.add_argument("distortion", nargs="?")
_ap.add_argument("--exclude", default="", help="comma list of test records to drop, e.g. 201")
_ap.add_argument("--out", default="analysis", help="output folder under results/")
_args = _ap.parse_args()
res_path = pathlib.Path(_args.results) if _args.results else ROOT / "results" / "test_results.csv"
dist_path = pathlib.Path(_args.distortion) if _args.distortion else ROOT / "results" / "test_distortion.csv"
EXCLUDE = [x for x in _args.exclude.split(",") if x]
out = ROOT / "results" / _args.out
out.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 230, "display.max_columns", 40, "display.max_rows", 300)

d = pd.read_csv(res_path, dtype={"record": str})
ref = d[d.method == "reference"].set_index("record")
d = d[d.method != "reference"].copy()
dl_path = ROOT / "results" / "dl_results.csv"
HAS_DL = dl_path.exists()
if HAS_DL:
    dl = pd.read_csv(dl_path, dtype={"record": str})
    d = pd.concat([d, dl[dl.method != "reference"]], ignore_index=True)
if EXCLUDE:
    d = d[~d.record.isin(EXCLUDE)].copy()
    ref = ref[~ref.index.isin(EXCLUDE)]
    print(f"EXCLUDING records {EXCLUDE}: {d.record.nunique()} records remain")
d["f1_ref"] = d["record"].map(ref["f1"])
d["rmssd_excess"] = d["rmssd_err_pct"] - d["record"].map(ref["rmssd_err_pct"])
RNG = np.random.default_rng(20261008)


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx]); adj[idx] = min(1.0, running)
    return adj


def paired(a, b, n_boot=10000):
    """a, b aligned per record. Wilcoxon signed-rank p, rank-biserial r, mean diff + bootstrap CI."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    nz = diff[diff != 0]
    if len(nz) == 0:
        p, r = 1.0, 0.0
    else:
        p = stats.wilcoxon(nz).pvalue if len(nz) >= 1 else 1.0
        ranks = stats.rankdata(np.abs(nz))
        r = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    idx = RNG.integers(0, len(diff), (n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    return {"n": len(diff), "mean_diff": diff.mean(), "ci_lo": np.percentile(boots, 2.5),
            "ci_hi": np.percentile(boots, 97.5), "median_diff": float(np.median(diff)),
            "rank_biserial": r, "p": p}


def wide(df, noise, snr, value):
    sub = df[(df.noise == noise) & (df.snr_target == snr)]
    return sub.pivot(index="record", columns="method", values=value)


# ---------------------------------------------------------------- T1 fidelity, T2 QRS
agg = lambda g, c: f"{g[c].mean():.2f} ± {g[c].std():.2f}"
t1 = (d.groupby(["noise", "snr_target", "method"])
        .apply(lambda g: pd.Series({"dSNR": agg(g, "delta_snr"), "RMSE_mV": f"{g.rmse.mean():.3f} ± {g.rmse.std():.3f}",
                                    "PRD_%": agg(g, "prd")}), include_groups=False).reset_index())
t1.to_csv(out / "table_fidelity.csv", index=False)
t2 = (d.groupby(["noise", "snr_target", "method"])
        .agg(F1=("f1", "mean"), Se=("se", "mean"), PPV=("ppv", "mean"), loc_ms=("loc_err_ms", "mean"),
             rmssd_err_pct=("rmssd_err_pct", "mean"), F1_ref=("f1_ref", "mean")).round(4).reset_index())
t2.to_csv(out / "table_qrs.csv", index=False)

# ---------------------------------------------------------------- E2 Friedman
rows = []
for (nt, snr), g in d[d.method != "none"].groupby(["noise", "snr_target"]):
    w = g.pivot(index="record", columns="method", values="delta_snr").dropna()
    if w.shape[1] >= 3:
        stat, p = stats.friedmanchisquare(*[w[c] for c in w.columns])
        rows.append({"noise": nt, "snr": snr, "n_records": len(w), "k_methods": w.shape[1],
                     "chi2": stat, "p": p, "best_mean": w.mean().idxmax()})
fr = pd.DataFrame(rows); fr["p_holm"] = holm(fr.p); fr.to_csv(out / "friedman_dsnr.csv", index=False)

# ---------------------------------------------------------------- P1
rows = []
for nt in ["bw", "em", "ma", "pli"]:
    for snr in sorted(d.snr_target.unique()):
        w = wide(d, nt, snr, "delta_snr")
        rows.append({"family": "P1", "noise": nt, "snr": snr, "contrast": "lms_oracle - rls_oracle",
                     **paired(w["lms_oracle"], w["rls_oracle"])})
p1 = pd.DataFrame(rows); p1["p_holm"] = holm(p1.p)

# ---------------------------------------------------------------- P2
rows = []
for snr in sorted(d.snr_target.unique()):
    w = wide(d, "pli", snr, "delta_snr")
    rows.append({"family": "P2", "noise": "pli", "snr": snr, "contrast": "lms_blind - butter_zp",
                 **paired(w["lms_blind"], w["butter_zp"])})
p2 = pd.DataFrame(rows); p2["p_holm"] = holm(p2.p)

# ---------------------------------------------------------------- P3 / E5 (DL)
p3 = e5 = pd.DataFrame()
if HAS_DL:
    rows = []
    for nt in ["bw", "em", "ma", "pli"]:
        for snr in sorted(d.snr_target.unique()):
            w = wide(d, nt, snr, "delta_snr").dropna(subset=["dl_unet"])
            rows.append({"family": "P3", "noise": nt, "snr": snr, "contrast": "dl_unet - butter_zp",
                         **paired(w["dl_unet"], w["butter_zp"])})
    p3 = pd.DataFrame(rows); p3["p_holm"] = holm(p3.p)
    rows = []
    for nt in ["bw", "em", "ma", "pli"]:
        others = {"bw": ["dwt", "median_bl"], "em": ["dwt"], "ma": ["dwt"],
                  "pli": ["dwt", "notch", "lms_blind", "rls_blind"]}[nt] + \
                 [("lms_oracle" if nt == "pli" else "rls_oracle")]
        for snr in sorted(d.snr_target.unique()):
            w = wide(d, nt, snr, "delta_snr").dropna(subset=["dl_unet"])
            for o in others:
                rows.append({"family": "E5", "noise": nt, "snr": snr, "contrast": f"dl_unet - {o}",
                             **paired(w["dl_unet"], w[o])})
    e5 = pd.DataFrame(rows); e5["p_holm"] = holm(e5.p)

# ---------------------------------------------------------------- P4
rows = []
for (nt, snr), g in d.groupby(["noise", "snr_target"]):
    w = g.pivot(index="record", columns="method", values="f1")
    for m in [c for c in w.columns if c != "none"]:
        rows.append({"family": "P4", "noise": nt, "snr": snr, "contrast": f"{m} - none (F1)", **paired(w[m], w["none"])})
p4 = pd.DataFrame(rows); p4["p_holm"] = holm(p4.p)

# ---------------------------------------------------------------- E1 oracle premium (PLI)
rows = []
for snr in sorted(d.snr_target.unique()):
    w = wide(d, "pli", snr, "delta_snr")
    for a, b in [("lms_oracle", "lms_blind"), ("rls_oracle", "rls_blind"), ("lms_blind", "notch"),
                 ("rls_blind", "notch"), ("lms_oracle", "notch")]:
        rows.append({"family": "E1", "noise": "pli", "snr": snr, "contrast": f"{a} - {b}", **paired(w[a], w[b])})
e1 = pd.DataFrame(rows); e1["p_holm"] = holm(e1.p)

rows = []
for (nt, snr), g in d.groupby(["noise", "snr_target"]):
    w = g.pivot(index="record", columns="method", values="rmssd_excess")
    for m in [c for c in w.columns if c != "none"]:
        ok = w[[m, "none"]].dropna()
        rows.append({"family": "E4", "noise": nt, "snr": snr, "contrast": f"{m} - none (RMSSD excess %-pts)",
                     **paired(ok[m], ok["none"])})
e4 = pd.DataFrame(rows); e4["p_holm"] = holm(e4.p)

allt = pd.concat([p1, p2, p3, p4, e1, e4, e5], ignore_index=True)
allt.to_csv(out / "paired_tests.csv", index=False)

# ---------------------------------------------------------------- E3 dSNR vs F1 gain
g = d[d.method != "none"].merge(d[d.method == "none"][["record", "noise", "snr_target", "f1"]],
                                on=["record", "noise", "snr_target"], suffixes=("", "_none"))
g["dF1"] = g.f1 - g.f1_none
rows = []
for nt, gg in g.groupby("noise"):
    rho, p = stats.spearmanr(gg.delta_snr, gg.dF1)
    rows.append({"noise": nt, "n": len(gg), "spearman_rho": rho, "p": p,
                 "share_dF1_nonzero": (gg.dF1 != 0).mean()})
e3 = pd.DataFrame(rows); e3.to_csv(out / "spearman_dsnr_vs_f1gain.csv", index=False)

# ---------------------------------------------------------------- E3b record-level Spearman
# Within each record and noise type: correlation across methods x input SNRs between delta-SNR gain and F1 gain
# (undefined when F1 does not change at all for that record/noise type). Summarised by the median over records.
rows = []
for (rec_, nt_), gg in g.groupby(["record", "noise"]):
    if gg.dF1.nunique() > 1 and gg.delta_snr.nunique() > 1:
        rows.append({"record": rec_, "noise": nt_, "n_obs": len(gg), "spearman_rho": stats.spearmanr(gg.delta_snr, gg.dF1).statistic})
e3b = pd.DataFrame(rows)
e3b.to_csv(out / "spearman_record_level_per_record.csv", index=False)
e3b_sum = e3b.groupby("noise").spearman_rho.agg(n_records_defined="count", median="median",
                                                q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)).reset_index()
e3b_sum["n_records_total"] = d.record.nunique()
e3b_sum.to_csv(out / "spearman_record_level_summary.csv", index=False)

# ---------------------------------------------------------------- filter-only distortion
if dist_path.exists():
    x = pd.read_csv(dist_path, dtype={"record": str})
    x = x[~x.record.isin(EXCLUDE)]
    dt = (x.groupby(["noise", "method"])["prd_noise_free"]
            .agg(["mean", "median", "max"]).round(2).reset_index())
    dt.to_csv(out / "distortion_noise_free.csv", index=False)

# ---------------------------------------------------------------- print
print(f"records: {d.record.nunique()}  | reference-ECG QRS F1 (no noise): {ref.f1.mean():.4f} (min {ref.f1.min():.3f})")
show = ["noise", "snr", "contrast", "n", "mean_diff", "ci_lo", "ci_hi", "rank_biserial", "p", "p_holm"]
for fam, t in (("P1 (planned): LMS vs RLS, oracle, dSNR", p1), ("P2 (secondary): lms_blind vs butter_zp, PLI, dSNR", p2),
               ("E1 (exploratory): oracle premium / blind vs notch, PLI, dSNR", e1)):
    print(f"\n=== {fam}\n", t[show].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
if HAS_DL:
    for fam, t in (("P3 (secondary): dl_unet vs butter_zp, dSNR", p3), ("E5 (exploratory): dl_unet vs other methods, dSNR", e5)):
        print(f"\n=== {fam}\n", t[show].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("\n=== P4 (planned): F1 vs no filtering - ONLY rows with Holm-p < 0.05")
sig = p4[p4.p_holm < 0.05]
print(sig[show].to_string(index=False, float_format=lambda v: f"{v:.3g}") if len(sig) else "none significant;", f"({len(sig)} of {len(p4)} tests)")
print("\n=== E4 (exploratory): RMSSD excess error vs no filtering - ONLY rows with Holm-p < 0.05")
sig4 = e4[e4.p_holm < 0.05]
print(sig4[show].to_string(index=False, float_format=lambda v: f"{v:.3g}") if len(sig4) else "none significant;", f"({len(sig4)} of {len(e4)} tests)")
print("\n=== Friedman on dSNR per condition\n", fr.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("\n=== E3: Spearman(dSNR gain, F1 gain over no filtering), POOLED over records, methods, SNRs\n", e3.round(4).to_string(index=False))
print("\n=== E3b: record-level Spearman (median over records in which F1 varies)\n", e3b_sum.round(3).to_string(index=False))
if dist_path.exists():
    print("\n=== filter-only distortion on reference ECG (PRD %, noise-free input)\n",
          dt[dt.noise.isin(['bw', 'pli'])].to_string(index=False))

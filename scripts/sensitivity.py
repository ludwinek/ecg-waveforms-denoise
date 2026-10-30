"""Sensitivity of the method comparison to the choice of REFERENCE ECG.

Inputs (all produced by run_test.py with the SAME frozen parameters):
  results/test_results.csv               main analysis (reference mode in config/protocol.yaml, spline_hp)
  results/test_results_spline.csv        --reference spline    (spline baseline removed, NO high-pass: independent of every filter)
  results/test_results_raw.csv           --reference raw       (raw MIT-BIH segment, DC removed only)
  results/test_results_highpass.csv      --reference highpass  (zero-phase 0.5 Hz high-pass)
Usage: python scripts/sensitivity.py
Outputs: results/analysis/sensitivity_reference.csv (+ printed summary).

What it answers
  (a) how much does delta-SNR of each method change with the reference?  The claim to check:
      methods that remove content below 0.5 Hz (Butterworth, median baseline, DWT) are penalised by raw references,
      methods that leave low frequencies alone (notch, oracle/blind adaptive filters) are not;
  (b) do the rankings and the confirmatory contrast P1 (LMS vs RLS, oracle) survive the change of reference?
The U-Net is NOT part of this analysis (it was trained against the main reference).
"""
import pathlib
import numpy as np, pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)

files = {"spline_hp (main)": RES / "test_results.csv", "spline only": RES / "test_results_spline.csv",
         "highpass 0.5 Hz": RES / "test_results_highpass.csv", "raw": RES / "test_results_raw.csv"}
data = {}
for k, f in files.items():
    if f.exists():
        d = pd.read_csv(f, dtype={"record": str})
        d = d[~d.method.isin(["reference", "none", "dl_unet"])]
        data[k] = d
    else:
        print(f"[missing] {f.name} - skipped")
if "spline_hp (main)" not in data or len(data) < 2:
    raise SystemExit("need results/test_results.csv and at least one of test_results_raw.csv / test_results_highpass.csv")
MAIN = "spline_hp (main)"
HP_TYPE = ["butter_zp", "median_bl", "dwt"]                       # remove content below ~0.5 Hz
LOW_KEEP = ["notch", "dwt_legacy", "lms_blind", "rls_blind", "lms_oracle", "rls_oracle"]   # leave low frequencies alone
# CANDIDATE blind methods: the six methods that compete as deployable blind denoisers. The legacy DWT (does almost nothing for
# BW/EM/MA) and the causal Butterworth filter are REFERENCE configurations and are not candidates. ALL_NONORACLE admits them as well.
BLIND = ["butter_zp", "median_bl", "notch", "dwt", "lms_blind", "rls_blind"]
ALL_NONORACLE = BLIND + ["butter_causal", "dwt_legacy"]
NAMES = {"butter_zp": "Butterworth", "butter_causal": "Butterworth (causal)", "median_bl": "Median baseline", "notch": "Notch",
         "dwt": "DWT (tuned)", "dwt_legacy": "DWT level 5 (legacy)", "lms_blind": "LMS, blind", "rls_blind": "RLS, blind",
         "lms_oracle": "LMS, oracle", "rls_oracle": "RLS, oracle"}

mean = {k: d.groupby(["noise", "snr_target", "method"]).delta_snr.mean() for k, d in data.items()}
tab = pd.concat(mean, axis=1)                                      # index (noise, snr, method), columns = references
tab.to_csv(RES / "analysis" / "sensitivity_reference.csv") if (RES / "analysis").exists() else (
    (RES / "analysis").mkdir(parents=True, exist_ok=True), tab.to_csv(RES / "analysis" / "sensitivity_reference.csv"))

# ---- (a) delta-SNR at 5 dB by method and reference
print("=== mean delta-SNR (dB) at 5 dB input SNR, by reference")
for nt in ["bw", "em", "ma", "pli"]:
    t = tab.xs((nt, 5), level=("noise", "snr_target")).round(1)
    t.index = [NAMES.get(m, m) for m in t.index]
    print(f"\n{nt.upper()}\n{t.to_string()}")

# ---- shift by method class
print("\n=== mean shift of delta-SNR relative to the main reference (dB, averaged over noise types and input SNRs)")
rows = []
for k in [c for c in tab.columns if c != MAIN]:
    sh = (tab[k] - tab[MAIN]).groupby(level="method").mean()
    rows.append({"reference": k, "high-pass-type (Butterworth, median, DWT)": sh.reindex(HP_TYPE).mean(),
                 "low-frequency-preserving (notch, adaptive, legacy DWT)": sh.reindex(LOW_KEEP).mean(),
                 **{NAMES[m]: sh.get(m, np.nan) for m in HP_TYPE + ["lms_oracle", "rls_oracle"]}})
print(pd.DataFrame(rows).round(2).to_string(index=False))

# ---- (b) rank agreement and winners
print("\n=== Spearman correlation of the method ranking (by mean delta-SNR) with the main reference, per noise type and SNR")
print("    methods in the ranking (all evaluated methods except no-filtering and the U-Net):",
      {nt: int(tab.xs(nt, level='noise').index.get_level_values('method').nunique()) for nt in ['bw', 'em', 'ma', 'pli']})
rows = []
for k in [c for c in tab.columns if c != MAIN]:
    for (nt, snr), g in tab.groupby(level=["noise", "snr_target"]):
        g = g.droplevel(["noise", "snr_target"])
        ok = g[[MAIN, k]].dropna()
        rho = stats.spearmanr(ok[MAIN], ok[k]).statistic if len(ok) > 2 else np.nan
        cand = [m for m in BLIND if m in ok.index]
        allno = [m for m in ALL_NONORACLE if m in ok.index]
        rows.append({"reference": k, "noise": nt, "snr": snr, "rho": rho,
                     "best_main": ok[MAIN].idxmax(), f"best_{k}": ok[k].idxmax(),
                     "best_blind_main": ok.loc[cand, MAIN].idxmax(), "best_blind_other": ok.loc[cand, k].idxmax(),
                     "best_nonoracle_main": ok.loc[allno, MAIN].idxmax(), "best_nonoracle_other": ok.loc[allno, k].idxmax()})
rk = pd.DataFrame(rows)
print(rk.groupby(["reference", "noise"]).rho.mean().round(2).unstack().to_string())
print("\nBest-method stability (conditions = 4 noise types x 4 SNR = 16 per reference)")
print(f"  candidate blind set = {BLIND}")
print(f"  all non-oracle set  = {ALL_NONORACLE}")
tot_c = tot_a = 0
for k in [c for c in tab.columns if c != MAIN]:
    sub = rk[rk.reference == k]
    ch_c = int((sub.best_blind_main != sub.best_blind_other).sum()); ch_a = int((sub.best_nonoracle_main != sub.best_nonoracle_other).sum())
    tot_c += ch_c; tot_a += ch_a
    print(f"  {k:16s}: best overall unchanged in {(sub.best_main == sub[f'best_{k}']).sum()} of {len(sub)}; "
          f"best CANDIDATE blind method changed in {ch_c}; best of ALL non-oracle methods changed in {ch_a}")
    if ch_a:
        print(sub[sub.best_nonoracle_main != sub.best_nonoracle_other][["noise", "snr", "best_nonoracle_main", "best_nonoracle_other"]].to_string(index=False))
print(f"  TOTAL over {3 * 16} comparisons: candidate set {tot_c}, all non-oracle set {tot_a}")

# ---- P1 under each reference
def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for r_, i in enumerate(o):
        run = max(run, (m - r_) * p[i]); adj[i] = min(1.0, run)
    return adj

print("\n=== P1 (LMS vs RLS, oracle reference; paired Wilcoxon, Holm over 16 tests) under each reference")
for k, d in data.items():
    rows = []
    for nt in ["bw", "em", "ma", "pli"]:
        for snr in sorted(d.snr_target.unique()):
            w = d[(d.noise == nt) & (d.snr_target == snr)].pivot(index="record", columns="method", values="delta_snr")
            diff = (w["lms_oracle"] - w["rls_oracle"]).dropna().values
            nz = diff[diff != 0]
            p = stats.wilcoxon(nz).pvalue if len(nz) else 1.0
            rows.append({"noise": nt, "snr": snr, "mean_diff": diff.mean(), "p": p})
    t = pd.DataFrame(rows); t["p_holm"] = holm(t.p)
    sign = {nt: ("RLS better" if (t[t.noise == nt].mean_diff < 0).all() else "LMS better" if (t[t.noise == nt].mean_diff > 0).all() else "mixed")
            for nt in ["bw", "em", "ma", "pli"]}
    print(f"{k:18s} direction per noise type: {sign};  all 16 Holm-significant: {bool((t.p_holm < 0.05).all())};  "
          f"|mean diff| range {t.mean_diff.abs().min():.1f}-{t.mean_diff.abs().max():.1f} dB")

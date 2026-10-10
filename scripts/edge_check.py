"""For every numeric tuned parameter with >=3 grid values: best mean delta-SNR at each value
(maximised over the other parameters), so you can see how much is left beyond a grid edge.
Usage: python scripts/edge_check.py   (reads results/tuning_dev.csv)"""
import ast, pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results" / "tuning_dev.csv")
for (m, nt), g in df.groupby(["method", "noise"]):
    sc = g.groupby("params")["delta_snr"].mean()
    P = pd.DataFrame([dict(ast.literal_eval(p)) for p in sc.index])
    P["score"] = sc.values
    for k in [c for c in P.columns if c != "score"]:
        sub = P
        if k == "thr_scale":                       # scale is meaningless when nothing is thresholded
            sub = P[P["thr_levels"].map(len) > 0]
        if sub[k].map(lambda v: isinstance(v, (tuple, list, str))).any() or sub[k].nunique() < 3 or sub.empty:
            continue
        best = sub.groupby(k)["score"].max().sort_index()
        steps = " | ".join(f"{idx}: {val:.2f}" for idx, val in best.items())
        edge = best.iloc[-1] - best.iloc[-2]; edge_lo = best.iloc[0] - best.iloc[1]
        flag = "  <-- still rising at upper edge" if best.idxmax() == best.index[-1] and edge > 0.05 else \
               "  <-- still rising at lower edge" if best.idxmax() == best.index[0] and edge_lo > 0.05 else ""
        print(f"{m:10s} {nt:3s} {k:9s} {steps}{flag}")

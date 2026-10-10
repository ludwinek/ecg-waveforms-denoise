"""Tune method parameters on the DEV records only, then freeze them.

Usage: python scripts/tune_dev.py data [--seconds 300] [--workers 4]

* Dev records: fixed by protocol seed (config/splits.yaml is written once and reused).
* Noise for tuning: NSTDB *train* window (0-1200 s); evaluation later uses the *test*
  window (1200-1800 s), so tuning never sees evaluation noise. PLI is synthetic.
* Fixed by standard (NOT tuned): Butterworth 0.5-40 Hz (order 4), notch (50 Hz, Q=30),
  median baseline (200/600 ms).
* Tuned: NLMS/RLS oracle (taps, mu / lambda), blind quadrature (mu / lambda),
  DWT (wavelet, approximation level, zero-detail levels).
* Criterion: mean delta-SNR over dev records and SNR levels (per method x noise type).
Outputs: config/splits.yaml, config/tuned_params.yaml, results/tuning_dev.csv
"""
import argparse, itertools, pathlib, sys
from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, yaml
from ecgdn import data, methods, noise
from ecgdn.reference import make_reference
from ecgdn.metrics import evaluate
from ecgdn.splits import make_splits

NT_REAL = ("bw", "em", "ma")


def grids():
    g = {}
    mus = (0.0001, 0.0003, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5)
    lms_taps = {"bw": (8, 16, 32, 64), "em": (2, 4, 8, 16, 32), "ma": (2, 4, 8, 16, 32), "pli": (16, 32, 64, 128)}
    g["lms_oracle"] = {nt: [dict(taps=t, mu=m) for t in lms_taps[nt] for m in mus] for nt in lms_taps}
    lams = (0.99, 0.995, 0.999, 0.9999, 0.99999, 0.999995, 1.0)
    rls_taps = {"bw": (4, 8, 16, 32), "em": (4, 8, 16, 32), "ma": (4, 8, 16, 32), "pli": (8, 16, 32, 64)}
    g["rls_oracle"] = {nt: [dict(taps=t, lam=l) for t in rls_taps[nt] for l in lams] for nt in rls_taps}
    g["lms_blind"] = {"pli": [dict(mu=m, harmonics=h) for m in (0.001, 0.005, 0.01, 0.05, 0.1)
                              for h in ((1,), (1, 3))]}
    g["rls_blind"] = {"pli": [dict(lam=l, harmonics=h) for l in (0.95, 0.98, 0.99, 0.995, 0.999, 0.9999)
                              for h in ((1,), (1, 3))]}
    def dwt_grid(scales):
        return [dict(wavelet=w, level=L, zero_details=zd, thr_levels=tl, mode=md, thr_scale=sc)
                for w in ("db4", "db6", "sym8") for L in (8, 9)
                for zd in ((), (1,), (2,), (1, 2))
                for tl in ((), (1, 2), (1, 2, 3), (1, 2, 3, 4, 5))
                for md in ("soft", "hard") for sc in scales
                if not (tl == () and (md == "hard" or sc != 1.0))]

    base = dwt_grid((0.3, 0.6, 1.0))
    # PLI: the optimum was still rising at scale 1.0 (edge_check) -> wider scale range
    g["dwt"] = {"bw": base, "em": base, "ma": base, "pli": dwt_grid((0.3, 0.6, 1.0, 1.5, 2.0, 3.0))}
    return g


def _key(o):
    return repr(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in o.items()))


def tune_record(args):
    rec, data_dir, cfg, seconds, only = args
    fs, seed = cfg["fs"], cfg["seed"]
    raw, _, ann_s, ann_sym = data.load_mitbih_segment(str(pathlib.Path(data_dir) / "mitdb"), rec,
                                                      cfg["channel_name"], seconds)
    ecg, _ = make_reference(raw, fs, ann_s, ann_sym, cfg["reference"]["mode"], cfg["reference"]["pr_offset_s"],
                              cfg["reference"]["spline_hp_cutoff_hz"], cfg["reference"]["highpass_cutoff_hz"])
    n = len(ecg)
    nstdb = {k: noise.load_nstdb_noise(str(pathlib.Path(data_dir) / "nstdb"), k)[0] for k in NT_REAL}
    G = grids()
    rows = []
    for ni, nt in enumerate(cfg["noise_types"]):
        for si, snr in enumerate(cfg["snr_in_db"]):
            rng = np.random.default_rng([seed, int(rec), ni, si])
            raw = (noise.synth_pli(n, fs, rng, **{k: cfg["pli"][k] for k in ("f0", "harmonic_amp", "freq_drift_hz", "am_depth")})
                   if nt == "pli" else
                   noise.nstdb_segment(nstdb[nt], fs, n, cfg["nstdb"]["train_seconds"], rng))
            noisy, v = noise.add_noise(ecg, raw, snr)
            ctx = {"noise_type": nt, "oracle_ref": noise.oracle_reference(v, cfg["oracle_reference"]["snr_ref_db"], rng)}
            for name, per_nt in G.items():
                if only and name not in only:
                    continue
                for ov in per_nt.get(nt, []):
                    out = methods.run(name, noisy, fs, ctx, ov)
                    m = evaluate(ecg, noisy, out, fs, cfg["warmup_seconds"], cfg["metrics"]["remove_dc"])
                    rows.append({"record": rec, "method": name, "noise": nt, "snr": snr,
                                 "params": _key(ov), "_ov": ov, "delta_snr": m["delta_snr"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data")
    ap.add_argument("--seconds", type=float, default=None, help="override segment length (testing only)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--methods", default=None, help="re-tune only these methods (comma list) and merge into existing files")
    ap.add_argument("--dev-limit", type=int, default=None, help="use only the first N dev records (testing only)")
    ap.add_argument("--reference", default=None, help="override reference.mode (spline|spline_hp|highpass|raw)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config" / "protocol.yaml"))
    if a.reference:
        cfg["reference"]["mode"] = a.reference
    print("reference mode:", cfg["reference"]["mode"])
    seconds = a.seconds or cfg["segment_seconds"]

    sp_path = ROOT / "config" / "splits.yaml"
    if sp_path.exists():
        sp = yaml.safe_load(open(sp_path))
    else:
        sp = make_splits(cfg["records_all"], cfg["excluded_records"], cfg["n_dev"], cfg["n_folds"], cfg["seed"])
        yaml.safe_dump(sp, open(sp_path, "w"))
    print("dev records:", sp["dev"])

    only = set(a.methods.split(",")) if a.methods else None
    dev = sp["dev"][:a.dev_limit] if a.dev_limit else sp["dev"]
    jobs = [(r, a.data_dir, cfg, seconds, only) for r in dev]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, part in enumerate(ex.map(tune_record, jobs), 1):
            rows += part
            print(f"  done {i}/{len(jobs)}", flush=True)

    df = pd.DataFrame(rows)
    ovmap = {r["params"]: r["_ov"] for r in rows}
    (ROOT / "results").mkdir(exist_ok=True)
    new_csv = df.drop(columns="_ov")
    csv_path = ROOT / "results" / "tuning_dev.csv"
    if only and csv_path.exists():
        old = pd.read_csv(csv_path)
        new_csv = pd.concat([old[~old.method.isin(only)], new_csv], ignore_index=True)
    new_csv.to_csv(csv_path, index=False)

    score = df.groupby(["method", "noise", "params"])["delta_snr"].mean().reset_index()
    G = grids()

    def at_edge(m, nt, ov):
        """Numeric parameters whose best value sits at the border of the searched grid."""
        edge = []
        for k, v in ov.items():
            if isinstance(v, (list, tuple)) or isinstance(v, str):
                continue
            vals = sorted({o[k] for o in G[m][nt] if k in o})
            if k == "lam" and v == 1.0:      # lambda = 1 is the natural upper bound (no forgetting)
                continue
            if len(vals) > 2 and v in (vals[0], vals[-1]):
                edge.append(k)
        return ",".join(edge)

    tuned_path = ROOT / "config" / "tuned_params.yaml"
    meta_path = ROOT / "config" / "tuned_meta.yaml"
    tuned, report = {}, []
    if only and tuned_path.exists():
        tuned = yaml.safe_load(open(tuned_path)) or {}
        if meta_path.exists() and yaml.safe_load(open(meta_path))["reference"] != cfg["reference"]:
            sys.exit("ABORT: existing tuned_params were made with a different reference; re-run without --methods.")
    for (m, nt), grp in score.groupby(["method", "noise"]):
        best = grp.sort_values("delta_snr", ascending=False).iloc[0]
        default = methods.REGISTRY[m][2]
        ov = {k: (list(v) if isinstance(v, tuple) else v) for k, v in ovmap[best.params].items()}
        tuned.setdefault(m, {})[nt] = ov
        report.append({"method": m, "noise": nt, "best_mean_dsnr": round(best.delta_snr, 2), "best_params": ov,
                       "grid_range": f"{grp.delta_snr.min():.2f}..{grp.delta_snr.max():.2f}",
                       "at_grid_edge": at_edge(m, nt, ovmap[best.params])})
    yaml.safe_dump(tuned, open(ROOT / "config" / "tuned_params.yaml", "w"), sort_keys=True)
    yaml.safe_dump({"reference": cfg["reference"], "segment_seconds": seconds, "dev_records": sp["dev"]},
                   open(ROOT / "config" / "tuned_meta.yaml", "w"))
    pd.set_option("display.width", 220, "display.max_colwidth", 80)
    print(pd.DataFrame(report).to_string(index=False))
    print("\nwrote config/tuned_params.yaml - FREEZE it (commit) before touching test records.")


if __name__ == "__main__":
    main()

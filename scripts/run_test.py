"""Evaluate all methods on the TEST records with FROZEN tuned parameters.

Usage: python scripts/run_test.py data [--workers 4] [--records 100,101] [--seconds 300]
Requires config/tuned_params.yaml + config/tuned_meta.yaml (from tune_dev.py) and config/splits.yaml.
Noise for evaluation comes from the NSTDB TEST window (1200-1800 s); PLI is synthetic.
Resumable: each finished record is saved to results/test_parts/; re-running skips them (use --force to redo).
Sensitivity: add --reference raw|highpass to score the same frozen parameters against another reference
(writes results/test_results_<mode>.csv; the main analysis is never touched).
Output: results/test_results.csv (one row per record x noise x SNR x method),
        results/test_distortion.csv (filter-only distortion per record x noise-type-tuned method)
"""
import argparse, hashlib, json, pathlib, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, yaml
from ecgdn import data, methods, noise
from ecgdn.experiment import filter_only_distortion, make_noisy
from ecgdn.metrics import compensate_delay, evaluate
from ecgdn.qrs import detect_qrs, qrs_scores
from ecgdn.reference import beat_positions, make_reference
from ecgdn.splits import make_splits

NT_REAL = ("bw", "em", "ma")


def overrides_for(tuned, nt):
    return {m: per[nt] for m, per in tuned.items() if nt in per}


def work(args):
    rec, data_dir, cfg, tuned, seconds, names = args
    fs, seed, R = cfg["fs"], cfg["seed"], cfg["reference"]
    raw, _, ann_s, ann_sym = data.load_mitbih_segment(str(pathlib.Path(data_dir) / "mitdb"), rec,
                                                      cfg["channel_name"], seconds)
    ref, _ = make_reference(raw, fs, ann_s, ann_sym, R["mode"], R["pr_offset_s"],
                            R["spline_hp_cutoff_hz"], R["highpass_cutoff_hz"])
    n = len(ref)
    w = int(round(cfg["warmup_seconds"] * fs))
    beats = beat_positions(ann_s, ann_sym, n)
    nstdb = {k: noise.load_nstdb_noise(str(pathlib.Path(data_dir) / "nstdb"), k)[0] for k in NT_REAL}
    skip_qrs = cfg.get("_skip_qrs", False)
    Q = (lambda sig: {}) if skip_qrs else (lambda sig: qrs_scores(beats, detect_qrs(sig, fs), fs, w, n - w))
    rows = [{"record": rec, "noise": "none", "snr_target": np.nan, "method": "reference", **Q(ref)}]
    drows = []
    for ni, nt in enumerate(cfg["noise_types"]):
        ov = overrides_for(tuned, nt)
        for m in names:
            if methods.applicable(m, nt) and "oracle" not in m:
                for d in filter_only_distortion(ref, fs, [m], cfg["warmup_seconds"], cfg["metrics"]["remove_dc"], ov):
                    drows.append({"record": rec, "noise": nt, **d})
        for si, snr in enumerate(cfg["snr_in_db"]):
            noisy, v, rng = make_noisy(ref, nt, snr, rec, ni, si, fs, cfg, nstdb)
            ctx = {"noise_type": nt, "oracle_ref": noise.oracle_reference(v, cfg["oracle_reference"]["snr_ref_db"], rng)}
            for m in names:
                if not methods.applicable(m, nt):
                    continue
                out = methods.run(m, noisy, fs, ctx, ov.get(m))
                if m in methods.CAUSAL:
                    out = compensate_delay(ref, out)
                rows.append({"record": rec, "noise": nt, "snr_target": snr, "method": m,
                             **evaluate(ref, noisy, out, fs, cfg["warmup_seconds"], cfg["metrics"]["remove_dc"]),
                             **Q(out)})
    return rows, drows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--records", default=None, help="comma list (testing only)")
    ap.add_argument("--seconds", type=float, default=None, help="segment length override (testing only)")
    ap.add_argument("--out", default="results/test_results.csv")
    ap.add_argument("--force", action="store_true", help="ignore saved per-record parts and recompute")
    ap.add_argument("--reference", default=None, choices=["spline", "spline_hp", "highpass", "raw"],
                    help="SENSITIVITY ANALYSIS: evaluate the FROZEN parameters against another reference mode "
                         "(separate output files; QRS detection skipped unless --qrs)")
    ap.add_argument("--qrs", action="store_true", help="with --reference: also run QRS detection")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config" / "protocol.yaml"))
    tuned = yaml.safe_load(open(ROOT / "config" / "tuned_params.yaml"))
    meta = yaml.safe_load(open(ROOT / "config" / "tuned_meta.yaml"))
    sens = a.reference is not None
    if sens:
        cfg["reference"]["mode"] = a.reference
        cfg["_skip_qrs"] = not a.qrs
        print(f"SENSITIVITY RUN: frozen parameters (tuned with {meta['reference']['mode']}) evaluated against reference '{a.reference}'")
    elif meta["reference"] != cfg["reference"]:
        sys.exit(f"ABORT: parameters were tuned with reference {meta['reference']}, protocol says {cfg['reference']}. Re-tune.")
    seconds = a.seconds or cfg["segment_seconds"]
    if a.seconds is None and meta["segment_seconds"] != seconds:
        sys.exit("ABORT: segment length differs from the one used for tuning.")
    sp_path = ROOT / "config" / "splits.yaml"
    sp = yaml.safe_load(open(sp_path)) if sp_path.exists() else make_splits(
        cfg["records_all"], cfg["excluded_records"], cfg["n_dev"], cfg["n_folds"], cfg["seed"])
    recs = a.records.split(",") if a.records else sp["test"]
    assert not set(recs) & set(sp["dev"]), "dev record in test set!"
    names = list(methods.REGISTRY)
    print(f"test records: {len(recs)}; reference mode: {cfg['reference']['mode']}")
    parts = ROOT / "results" / ("test_parts" if not sens else f"test_parts_{a.reference}")
    parts.mkdir(parents=True, exist_ok=True)
    fp = hashlib.sha256(json.dumps({"ref": cfg["reference"], "tuned": tuned, "seconds": seconds, "names": names,
                                    "snr": cfg["snr_in_db"], "noise": cfg["noise_types"], "seed": cfg["seed"],
                                    "skip_qrs": cfg.get("_skip_qrs", False)},
                                   sort_keys=True, default=str).encode()).hexdigest()
    fp_file = parts / "_fingerprint.txt"
    if fp_file.exists() and fp_file.read_text() != fp and not a.force:
        sys.exit("ABORT: saved parts were made with a different configuration/tuned parameters. "
                 "Delete results/test_parts or use --force.")
    if a.force:
        for f in parts.glob("*.csv"):
            f.unlink()
    fp_file.write_text(fp)
    todo = [r for r in recs if not (parts / f"{r}.csv").exists()]
    print(f"already done: {len(recs) - len(todo)}; to compute: {len(todo)}")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, (rec, a.data_dir, cfg, tuned, seconds, names)): rec for rec in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = futs[fut]
            try:
                r, d = fut.result()
            except Exception as e:
                print(f"  FAILED {rec}: {type(e).__name__}: {e}", flush=True)
                continue
            pd.DataFrame(d).to_csv(parts / f"{rec}_dist.csv", index=False)
            pd.DataFrame(r).to_csv(parts / f"{rec}.csv", index=False)   # written last = "record complete"
            print(f"  done {rec} ({i}/{len(todo)})", flush=True)
    missing = [r for r in recs if not (parts / f"{r}.csv").exists()]
    if missing:
        sys.exit(f"INCOMPLETE - missing records: {missing}. Fix the error above and re-run to resume.")
    out_name = a.out if not (sens and a.out == "results/test_results.csv") else f"results/test_results_{a.reference}.csv"
    out = ROOT / out_name
    out.parent.mkdir(exist_ok=True)
    pd.concat([pd.read_csv(parts / f"{r}.csv", dtype={"record": str}) for r in recs], ignore_index=True).to_csv(out, index=False)
    pd.concat([pd.read_csv(parts / f"{r}_dist.csv", dtype={"record": str}) for r in recs], ignore_index=True).to_csv(
        out.with_name("test_distortion.csv" if not sens else f"test_distortion_{a.reference}.csv"), index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()

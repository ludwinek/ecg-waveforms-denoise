"""Sanity-check the downloaded databases against the protocol. Usage: python scripts/check_data.py data"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, yaml
import wfdb
from ecgdn import data, noise

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data")
cfg = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1] / "config" / "protocol.yaml"))
rows, problems = [], []

for rec in cfg["records_all"]:
    path = root / "mitdb" / rec
    try:
        h = wfdb.rdheader(str(path))
        ok_ch = cfg["channel_name"] in h.sig_name
        x, fs, ann_s, ann_sym = data.load_mitbih_segment(str(root / "mitdb"), rec, cfg["channel_name"],
                                                         cfg["segment_seconds"])
        rows.append({"record": rec, "channels": "/".join(h.sig_name), "fs": fs, "n_total": h.sig_len,
                     "seg_samples": len(x), "seg_mean_mV": round(float(x.mean()), 3),
                     "seg_std_mV": round(float(x.std()), 3), "n_ann_seg": len(ann_s),
                     "excluded": rec in cfg["excluded_records"]})
        if not ok_ch: problems.append(f"{rec}: no {cfg['channel_name']}")
        if fs != cfg["fs"]: problems.append(f"{rec}: fs={fs}")
        if len(x) != int(cfg["segment_seconds"] * fs): problems.append(f"{rec}: short segment {len(x)}")
        if not np.isfinite(x).all(): problems.append(f"{rec}: non-finite samples")
    except KeyError as e:
        if rec in cfg["excluded_records"]:
            print(f"note: {rec} is excluded by protocol and has no {cfg['channel_name']} ({e}) - OK")
        else:
            problems.append(f"{rec}: KeyError: {e}")
    except Exception as e:
        problems.append(f"{rec}: {type(e).__name__}: {e}")

nst = []
for kind in ("bw", "em", "ma"):
    try:
        x, fs = noise.load_nstdb_noise(str(root / "nstdb"), kind)
        nst.append({"noise": kind, "fs": fs, "seconds": round(len(x) / fs, 1), "std_mV": round(float(x.std()), 3)})
        if fs != cfg["fs"]: problems.append(f"nstdb {kind}: fs={fs}")
        if len(x) / fs < cfg["nstdb"]["test_seconds"][1]:
            problems.append(f"nstdb {kind}: shorter than {cfg['nstdb']['test_seconds'][1]} s")
    except Exception as e:
        problems.append(f"nstdb {kind}: {type(e).__name__}: {e}")

pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
df = pd.DataFrame(rows)
(root.parent / "results").mkdir(exist_ok=True)
df.to_csv(root.parent / "results" / "data_check.csv", index=False)
print(df.to_string(index=False)); print("\nNSTDB:\n", pd.DataFrame(nst).to_string(index=False))
usable = int((~df.excluded).sum()) if len(df) else 0
print(f"\nrecords loaded: {len(df)}/{len(cfg['records_all'])}; usable after exclusions: {usable} (expected {len(cfg['records_all']) - len(cfg['excluded_records'])})")
if usable != len(cfg['records_all']) - len(cfg['excluded_records']): problems.append("unexpected number of usable records")
print("PROBLEMS:" if problems else "No problems found.")
for p in problems: print(" -", p)
sys.exit(1 if problems else 0)

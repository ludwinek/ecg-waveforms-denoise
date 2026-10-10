"""Download MIT-BIH Arrhythmia DB and MIT-BIH Noise Stress Test DB from PhysioNet (needs internet)."""
import pathlib, sys
import wfdb

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data")
for name, sub in (("mitdb", "mitdb"), ("nstdb", "nstdb")):
    out = root / sub
    out.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(name, str(out))
    print("downloaded", name, "->", out)

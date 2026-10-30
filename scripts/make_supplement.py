"""Assemble the supplementary material of the paper in ./supplementary (to be committed to the repository).

Run first (all from the repository root):
    python scripts/analyze.py
    python scripts/analyze.py --exclude 201 --out analysis_excl201
    python scripts/run_test.py data --workers 4 --reference raw          # also: spline, highpass
    python scripts/sensitivity.py
then:
    python scripts/make_supplement.py

Result: supplementary/S1..S7 (CSV), supplementary/README.txt and supplementary/results/ (per-record results the tables are computed from;
results/ itself is git-ignored). Nothing is recomputed here, files are only copied and renamed.
"""
import pathlib, shutil, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "supplementary"
TABLES = [  # (name in supplementary/, source, required)
    ("S1_fidelity_all_conditions_mean_sd.csv", RES / "analysis" / "table_fidelity.csv", True),
    ("S2_qrs_rr_rmssd_all_conditions.csv", RES / "analysis" / "table_qrs.csv", True),
    ("S3_paired_tests_raw_and_Holm_p.csv", RES / "analysis" / "paired_tests.csv", True),
    ("S4_paired_tests_without_record_201.csv", RES / "analysis_excl201" / "paired_tests.csv", True),
    ("S5_filter_distortion_noise_free.csv", RES / "analysis" / "distortion_noise_free.csv", True),
    ("S6_spearman_record_level_summary.csv", RES / "analysis" / "spearman_record_level_summary.csv", True),
    ("S6b_spearman_record_level_per_record.csv", RES / "analysis" / "spearman_record_level_per_record.csv", True),
    ("S7_sensitivity_reference.csv", RES / "analysis" / "sensitivity_reference.csv", False),
]
PER_RECORD = ["test_results.csv", "test_distortion.csv", "dl_results.csv",
              "test_results_raw.csv", "test_results_spline.csv", "test_results_highpass.csv"]

OUT.mkdir(exist_ok=True)
(OUT / "results").mkdir(exist_ok=True)
missing_required = []
for name, src, required in TABLES:
    if src.exists():
        shutil.copy2(src, OUT / name); print(f"ok       {name}")
    else:
        print(f"MISSING  {name}   <- {src.relative_to(ROOT)}")
        if required: missing_required.append(name)
for f in PER_RECORD:
    if (RES / f).exists():
        shutil.copy2(RES / f, OUT / "results" / f); print(f"ok       results/{f}")
    else:
        print(f"missing  results/{f}")
shutil.copy2(pathlib.Path(__file__).with_name("supplementary_README.txt"), OUT / "README.txt")
size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
print(f"\nsupplementary/ written ({size:.1f} MB)")
if missing_required:
    sys.exit("ABORT: required tables missing - run analyze.py first: " + ", ".join(missing_required))

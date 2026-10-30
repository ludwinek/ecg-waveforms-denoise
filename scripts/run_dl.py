"""Train and evaluate the 1D U-Net denoiser with 4-fold RECORD-WISE cross-validation.

Usage
  python scripts/run_dl.py data --smoke          # ~minutes: 1 fold, 2 test records, tiny training - run this FIRST
  python scripts/run_dl.py data                  # full run (4 folds)
  python scripts/run_dl.py data --folds 0,1      # subset of folds (resumable: finished folds are skipped)

Protocol
  * test records = the 36 test records; folds come from config/splits.yaml (4 x 9 records);
    for fold k the model is trained on the other 27 test records (minutes 5-30 of each record,
    i.e. never on the evaluated first 5 minutes) and evaluated on the 9 held-out records.
  * validation / early stopping: dev records except 202 (202 and 201 are the same patient).
  * training noise: NSTDB TRAIN window only + synthetic PLI; evaluation noise: NSTDB TEST window.
  * evaluation inputs come from experiment.make_noisy, i.e. they are bit-identical to what the
    classical/adaptive methods received in run_test.py.
Output: results/dl_results.csv (same schema as test_results.csv, method = dl_unet)
"""
import argparse, hashlib, json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, yaml
from ecgdn import data, noise
from ecgdn.dl_data import NoiseBank, TrainSampler, build_pli_pool, denoise_signal
from ecgdn.experiment import NT_REAL, make_noisy
from ecgdn.metrics import evaluate
from ecgdn.qrs import detect_qrs, qrs_scores
from ecgdn.reference import beat_positions, make_reference

METHOD = "dl_unet"
WIN, HOP = 720, 360


def build_ref(mitdb, rec, cfg, seconds, start_s=0.0):
    R, fs = cfg["reference"], cfg["fs"]
    raw, _, ann_s, ann_sym = data.load_mitbih_segment(mitdb, rec, cfg["channel_name"], seconds, start_s)
    ref, _ = make_reference(raw, fs, ann_s, ann_sym, R["mode"], R["pr_offset_s"],
                            R["spline_hp_cutoff_hz"], R["highpass_cutoff_hz"])
    return ref, beat_positions(ann_s, ann_sym, len(ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data")
    ap.add_argument("--folds", default="0,1,2,3")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-start", type=float, default=300.0)
    ap.add_argument("--train-seconds", type=float, default=1500.0)
    ap.add_argument("--eval-seconds", type=float, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check; writes results/dl_smoke.csv")
    ap.add_argument("--stub", action="store_true", help="identity model, no torch (pipeline test only)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config" / "protocol.yaml"))
    sp = yaml.safe_load(open(ROOT / "config" / "splits.yaml"))
    fs = cfg["fs"]
    folds = [int(f) for f in a.folds.split(",")]
    eval_seconds = a.eval_seconds or cfg["segment_seconds"]
    if a.smoke:
        folds, a.epochs, a.steps, a.bs, a.patience = [folds[0]], 2, 20, 64, 2
        eval_seconds, a.train_seconds = 60.0, 120.0
    w = int(round(cfg["warmup_seconds"] * fs))
    mitdb = str(pathlib.Path(a.data_dir) / "mitdb")
    nstdb_dir = str(pathlib.Path(a.data_dir) / "nstdb")
    torch = None
    if not a.stub:
        import torch
        from ecgdn import dl_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "(CPU - will be slow)")

    nst_full = {k: noise.load_nstdb_noise(nstdb_dir, k)[0] for k in NT_REAL}
    t0, t1 = cfg["nstdb"]["train_seconds"]
    nst_train = {k: v[int(t0 * fs):int(t1 * fs)] for k, v in nst_full.items()}
    bank = NoiseBank(nst_train, build_pli_pool(fs, seed=a.seed))

    parts = ROOT / "results" / ("dl_smoke_parts" if a.smoke else "dl_parts")
    parts.mkdir(parents=True, exist_ok=True)
    fp = hashlib.sha256(json.dumps({"a": {k: v for k, v in vars(a).items() if k not in ("folds", "force")},
                                    "ref": cfg["reference"], "folds": sp["folds"], "seed": cfg["seed"]},
                                   sort_keys=True, default=str).encode()).hexdigest()
    fp_file = parts / "_fingerprint.txt"
    if fp_file.exists() and fp_file.read_text() != fp and not a.force:
        sys.exit("ABORT: saved folds were made with different settings. Delete the parts folder or use --force.")
    if a.force:
        for f in parts.glob("*.csv"):
            f.unlink()
    fp_file.write_text(fp)

    # validation set: dev records except 202 (same patient as test record 201)
    val_recs = [r for r in sp["dev"] if r != "202"]
    if not a.stub:
        val_refs = [build_ref(mitdb, r, cfg, min(300.0, a.train_seconds), a.train_start)[0] for r in val_recs]
        val_x, val_y = TrainSampler(val_refs, bank).batch(2048 if a.smoke else 4096, np.random.default_rng(a.seed + 999))

    (ROOT / "models").mkdir(exist_ok=True)
    for k in folds:
        part = parts / f"fold{k}.csv"
        if part.exists():
            print(f"fold {k}: already done"); continue
        test_recs = sp["folds"][k][:2] if a.smoke else sp["folds"][k]
        train_recs = [r for r in sp["test"] if r not in sp["folds"][k]]
        assert not set(train_recs) & set(sp["folds"][k]) and not set(train_recs) & set(sp["dev"])
        print(f"fold {k}: train on {len(train_recs)} records, test on {test_recs}", flush=True)
        t_start = time.time()
        if a.stub:
            predict = lambda x: x
        else:
            refs = [build_ref(mitdb, r, cfg, a.train_seconds, a.train_start)[0] for r in train_recs]
            sampler = TrainSampler(refs, bank)
            model, best = dl_model.train_model(sampler, val_x, val_y, device, a.epochs, a.steps, a.bs, a.lr,
                                               a.patience, a.base, a.seed + k)
            torch.save(model.state_dict(), ROOT / "models" / f"dl_fold{k}{'_smoke' if a.smoke else ''}.pt")
            print(f"  trained in {time.time() - t_start:.0f} s, best val {best:.4f}, params {dl_model.n_params(model):,}")
            predict = dl_model.make_predictor(model, device)

        rows = []
        for rec in test_recs:
            ref, beats = build_ref(mitdb, rec, cfg, eval_seconds)
            n = len(ref)
            for ni, nt in enumerate(cfg["noise_types"]):
                for si, snr in enumerate(cfg["snr_in_db"]):
                    noisy, _, _ = make_noisy(ref, nt, snr, rec, ni, si, fs, cfg, nst_full)
                    out = denoise_signal(predict, noisy, WIN, HOP)
                    rows.append({"record": rec, "noise": nt, "snr_target": snr, "method": METHOD,
                                 **evaluate(ref, noisy, out, fs, cfg["warmup_seconds"], cfg["metrics"]["remove_dc"]),
                                 **qrs_scores(beats, detect_qrs(out, fs), fs, w, n - w)})
            print(f"  evaluated {rec}", flush=True)
        pd.DataFrame(rows).to_csv(part, index=False)

    done = [parts / f"fold{k}.csv" for k in range(len(sp["folds"])) if (parts / f"fold{k}.csv").exists()]
    out_path = ROOT / "results" / ("dl_smoke.csv" if a.smoke else "dl_results.csv")
    if a.smoke or len(done) == len(sp["folds"]):
        pd.concat([pd.read_csv(f, dtype={"record": str}) for f in done], ignore_index=True).to_csv(out_path, index=False)
        print("wrote", out_path)
    else:
        print(f"{len(done)}/{len(sp['folds'])} folds done - run again to finish (results/dl_results.csv is written when all 4 exist)")


if __name__ == "__main__":
    main()

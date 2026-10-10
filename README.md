# ecgdn - controlled comparison of ECG denoising methods

Classical (Butterworth, notch, median baseline, DWT), adaptive (NLMS, RLS; oracle and blind
reference) and, later, a small DL denoiser, evaluated on MIT-BIH with real (NSTDB) and extended
synthetic noise. Metrics: delta-SNR, RMSE, PRD, then QRS detection and RR/HRV (to be added).

## Quick start
    pip install -r requirements.txt
    pip install -e .
    pytest                              # unit tests (synthetic data, no downloads)
    python scripts/smoke_synthetic.py   # end-to-end plumbing check (NOT a result)
    python scripts/download_data.py data   # MIT-BIH + NSTDB from PhysioNet

`config/protocol.yaml` is the frozen protocol (seed, SNR levels, records, split sizes).

## Conventions that matter
* DC is removed before comparing signals (metrics) and before adaptation (NLMS/RLS "AC coupling").
  Without it, filters with a high-pass edge and adaptive cancellers are scored/behave wrongly.
* DWT: at fs=360 Hz the level-L approximation spans 0..fs/2^(L+1) Hz (L=8: 0.70 Hz, L=9: 0.35 Hz).
  Thresholding keeps large coefficients, so strong narrow-band interference needs explicit removal.
* "oracle" adaptive methods use a noisy copy of the TRUE noise as reference: an upper bound only.
* `synth*` / `synth_*_like` exist for tests only. Never report numbers obtained with them.

## Reference ECG (important)
Raw MIT-BIH segments contain baseline drift (a quarter of the AC power or more lies below 0.5 Hz),
so scoring against the raw record caps the SNR of any high-pass filter at ~2.5-6 dB and rewards
methods that keep the drift. `reference.make_reference` builds a drift-corrected reference
(`spline` = natural cubic spline through PR-segment knots, independent of all compared filters;
`highpass` = sensitivity analysis; `raw` = reporting only). Metrics cut `warmup_seconds` at BOTH ends.
Run `python scripts/diagnose.py data` before tuning; re-tune whenever the reference mode changes.

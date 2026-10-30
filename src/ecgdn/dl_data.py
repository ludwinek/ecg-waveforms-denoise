"""Data side of the DL denoiser (numpy only, unit-testable without PyTorch).

Training pairs are generated ON THE FLY from drift-corrected reference ECG + noise:
  * real NSTDB noise (bw/em/ma) taken ONLY from the TRAIN window (default 0-1200 s) -
    the evaluation noise (1200-1800 s) is never seen;
  * extended synthetic PLI (random harmonic amplitude / drift / AM depth);
  * input SNR ~ U(snr_lo, snr_hi) per window; noise type drawn uniformly -> the model is BLIND
    to noise type and SNR, like Butterworth/DWT.
Windows are NOT mean-removed (that would delete the reference's own window mean and create an
artificial distortion floor); input and target are divided by the std of the NOISY window,
information that is available at inference time. The reference is drift-corrected, so the
isoelectric line is near zero and the offsets introduced by noise must be removed by the network.
"""
from __future__ import annotations

import numpy as np

from .noise import synth_pli

KINDS = ("bw", "em", "ma", "pli")


def build_pli_pool(fs, n_series=30, seconds=600, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    pool = []
    for _ in range(n_series):
        pool.append(synth_pli(n, fs, rng, f0=50.0, harmonic_amp=rng.uniform(0.0, 0.4),
                              freq_drift_hz=rng.uniform(0.05, 0.5), am_depth=rng.uniform(0.0, 0.4)))
    return np.stack(pool)


class NoiseBank:
    def __init__(self, nstdb_train: dict, pli_pool: np.ndarray):
        self.real = {k: np.asarray(v, float) for k, v in nstdb_train.items()}
        self.pli = pli_pool

    def sample(self, kind, win, rng):
        a = self.pli[rng.integers(len(self.pli))] if kind == "pli" else self.real[kind]
        s = int(rng.integers(0, len(a) - win + 1))
        return a[s:s + win]


class TrainSampler:
    def __init__(self, refs, bank: NoiseBank, win=720, snr_range=(-2.0, 22.0), kinds=KINDS, min_var=1e-3):
        self.refs = [np.asarray(r, float) for r in refs]
        self.bank, self.win, self.snr_range, self.kinds, self.min_var = bank, win, snr_range, kinds, min_var
        lens = np.array([len(r) for r in self.refs], float)
        self.p = lens / lens.sum()

    def batch(self, bs, rng):
        w = self.win
        X = np.empty((bs, 1, w), np.float32)
        Y = np.empty((bs, 1, w), np.float32)
        i = 0
        while i < bs:
            r = self.refs[rng.choice(len(self.refs), p=self.p)]
            s = int(rng.integers(0, len(r) - w + 1))
            y = r[s:s + w]
            if float(np.var(y)) < self.min_var:       # flat / lead-off segment: no ECG to learn from
                continue
            py = float(np.mean(y ** 2))
            v = self.bank.sample(self.kinds[int(rng.integers(len(self.kinds)))], w, rng)
            pv = float(np.mean(v ** 2))
            if pv <= 0:
                continue
            snr = rng.uniform(*self.snr_range)
            noisy = y + v * np.sqrt(py / (pv * 10.0 ** (snr / 10.0)))
            sd = noisy.std() + 1e-8
            X[i, 0] = noisy / sd
            Y[i, 0] = y / sd
            i += 1
        return X, Y


# ----------------------------------------------------------------------------- inference
def window_starts(n, win, hop):
    st = list(range(0, n - win + 1, hop))
    if st[-1] != n - win:
        st.append(n - win)
    return np.array(st)


def denoise_signal(model_fn, sig, win=720, hop=360, batch=512):
    """Sliding-window inference with Hann-weighted overlap-add.

    model_fn: (B,1,win) float32 -> (B,1,win) float32 (normalised domain).
    Each window is divided by its own noisy std (not mean-removed), as in training. The global
    mean of the signal is removed first.
    """
    sig = np.asarray(sig, float)
    sig = sig - sig.mean()
    n = len(sig)
    st = window_starts(n, win, hop)
    wts = np.hanning(win + 2)[1:-1]
    num = np.zeros(n)
    den = np.zeros(n)
    for b in range(0, len(st), batch):
        idx = st[b:b + batch]
        seg = np.stack([sig[s:s + win] for s in idx])
        sd = seg.std(axis=1, keepdims=True) + 1e-8
        out = np.asarray(model_fn((seg / sd)[:, None, :].astype(np.float32)))[:, 0, :].astype(float) * sd
        for s, o in zip(idx, out):
            num[s:s + win] += wts * o
            den[s:s + win] += wts
    return num / den

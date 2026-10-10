"""Record-wise dev/test split and cross-validation folds (deterministic)."""
from __future__ import annotations

import numpy as np


def make_splits(records_all, excluded, n_dev, n_folds, seed):
    recs = sorted(r for r in records_all if r not in set(excluded))
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(recs))
    dev = sorted(perm[:n_dev])
    test = perm[n_dev:]
    folds = [sorted(map(str, test[i::n_folds])) for i in range(n_folds)]
    return {"dev": [str(r) for r in dev], "test": sorted(map(str, test)), "folds": folds}

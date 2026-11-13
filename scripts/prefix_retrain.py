from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from per_observation_baseline import (load_regime, fit_gb, evaluate,
                                      matched_fpr_table, prefix_vec,
                                      THRESHOLDS, RNG_SEED, N_FOLDS, TRAIN_N)


def precompute(runs):
    cache = {}
    for i, r in enumerate(runs):
        total = max(len(r["steps"]), 1)
        cache[r["run_id"]] = np.array([prefix_vec(r, N) for N in range(1, total + 1)])
        if (i + 1) % 500 == 0:
            print(f"    cached {i+1}/{len(runs)}")
    return cache


def label(run):
    return 1 if run["outcome"] == "unsafe" else 0


def traces_allN(train, test, cache):
    X = np.vstack([cache[r["run_id"]] for r in train])
    y = np.concatenate([np.full(len(cache[r["run_id"]]), label(r)) for r in train])
    clf, scaler = fit_gb(X, y)
    out = {}
    for r in test:
        M = cache[r["run_id"]]
        p = clf.predict_proba(scaler.transform(M))[:, 1]
        out[r["run_id"]] = {N: float(p[N - 1]) for N in range(1, len(M) + 1)}
    return out


def traces_perN(train, test, cache, max_n):
    models = {}
    for N in range(1, max_n + 1):
        Xs, ys = [], []
        for r in train:
            M = cache[r["run_id"]]
            if len(M) >= N:
                Xs.append(M[N - 1])
                ys.append(label(r))
        if len(set(ys)) < 2 or len(ys) < 40:
            models[N] = None
            continue
        models[N] = fit_gb(np.array(Xs), np.array(ys))

    def model_for(N):
        for k in range(N, 0, -1):
            if models.get(k) is not None:
                return models[k]
        return None

    out = {}
    for r in test:
        M = cache[r["run_id"]]
        trace = {}
        for N in range(1, len(M) + 1):
            m = model_for(min(N, max_n))
            if m is None:
                trace[N] = 0.0
                continue
            clf, scaler = m
            trace[N] = float(clf.predict_proba(scaler.transform(M[N - 1:N]))[0, 1])
        out[r["run_id"]] = trace
    return out


def main():
    out = {"meta": {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "seed": RNG_SEED, "train_N_reference": TRAIN_N}}
    for regime, tag in (("a", "ciphertext"), ("b", "wordcodec_A")):
        print(f"\n=== {tag} ===")
        runs = load_regime(regime)
        print("  precomputing prefix features ...")
        cache = precompute(runs)
        max_n = max(len(cache[r["run_id"]]) for r in runs)
        print(f"  longest run = {max_n} steps")

        y = np.array([label(r) for r in runs])
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
        folds = list(skf.split(np.zeros((len(runs), 1)), y))

        traces_by_det = {}
        for name, fn in (("prefix-GB-allN", traces_allN),
                         ("prefix-GB-perN", traces_perN)):
            t0 = time.time()
            tr = {}
            for tr_idx, te_idx in folds:
                a = [runs[i] for i in tr_idx]
                b = [runs[i] for i in te_idx]
                tr.update(fn(a, b, cache) if fn is traces_allN
                          else fn(a, b, cache, max_n))
            traces_by_det[name] = tr
            print(f"  {name:18s} {time.time()-t0:5.0f}s")

        block = {"detectors": {n: evaluate(runs, t, THRESHOLDS, n)
                               for n, t in traces_by_det.items()},
                 "matched_fpr": matched_fpr_table(runs, traces_by_det)}
        out[tag] = block
        with open(os.path.join(HERE, "..", "results", "prefix_retrain.json"), "w") as f:
            json.dump(out, f, indent=1)
    print("\nDONE")


if __name__ == "__main__":
    main()

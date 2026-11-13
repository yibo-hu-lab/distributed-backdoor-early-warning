from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from per_observation_baseline import load_regime, fit_gb, N_FOLDS
from fair_comparison import (PREFIX_VIEWS, PEROBS_VIEWS, prefix_cache,
                             perobs_cache, label)

SEED = 42
BUDGET = 0.01
ROWS = [("prefix_len", "Trajectory prefix", "available"),
        ("prefix_nolen", "Trajectory prefix", "withheld"),
        ("perobs_len", "Per-observation", "available"),
        ("perobs_nolen", "Per-observation", "withheld")]


def oof_traces(runs, cache, seed):
    y = np.array([label(r) for r in runs])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    traces = {}
    for tr_idx, te_idx in skf.split(np.zeros((len(runs), 1)), y):
        tr = [runs[i] for i in tr_idx]
        X = np.vstack([cache[r["run_id"]] for r in tr])
        yy = np.concatenate([np.full(len(cache[r["run_id"]]), label(r)) for r in tr])
        clf, sc = fit_gb(X, yy)
        peaks = sorted((float(clf.predict_proba(sc.transform(cache[r["run_id"]]))[:, 1].max())
                        for r in tr if r["_stratum"] == "clean"), reverse=True)
        k = int(np.floor(BUDGET * len(peaks)))
        th = np.nextafter(peaks[0], np.inf) if k <= 0 else peaks[min(k, len(peaks)) - 1]
        for i in te_idx:
            r = runs[i]
            p = clf.predict_proba(sc.transform(cache[r["run_id"]]))[:, 1]
            traces[r["run_id"]] = next((N for N, v in enumerate(p, 1) if v >= th), None)
    return traces


def main():
    out = {}
    for regime, tag in (("a", "ciphertext"), ("b", "word-codec")):
        runs = load_regime(regime)
        caches = {n: prefix_cache(runs, f) for n, f in PREFIX_VIEWS.items()}
        caches.update({n: perobs_cache(runs, f) for n, f in PEROBS_VIEWS.items()})
        unsafe = [r for r in runs if r["_stratum"] == "unsafe"]
        out[tag] = {}
        for key, _, _ in ROWS:
            first = oof_traces(runs, caches[key], SEED)
            hits = [first[r["run_id"]] for r in unsafe if first[r["run_id"]] is not None]
            out[tag][key] = {"detection_rate": len(hits) / len(unsafe),
                             "median_first_N": float(np.median(hits)) if hits else None,
                             "n_unsafe": len(unsafe)}
            print(f"{tag:11s} {key:13s} {100 * out[tag][key]['detection_rate']:5.1f}%  "
                  f"N*={out[tag][key]['median_first_N']}", flush=True)

    print("\nTable 4")
    print(f"{'View':18s} {'Length':10s} {'Ciphertext':>14s} {'Word-codec':>14s}")
    for key, view, length in ROWS:
        c, w = out["ciphertext"][key], out["word-codec"][key]
        print(f"{view:18s} {length:10s} {100*c['detection_rate']:6.1f}% N*={c['median_first_N']:<3g}"
              f" {100*w['detection_rate']:6.1f}% N*={w['median_first_N']:<3g}")
    os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
    with open(os.path.join(HERE, "..", "results", "table4.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

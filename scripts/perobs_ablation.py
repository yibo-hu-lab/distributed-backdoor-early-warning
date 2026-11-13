from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import per_observation_baseline as base
from per_observation_baseline import (load_regime, fit_gb, evaluate,
                                      matched_fpr_table, step_is_injected,
                                      shannon_entropy, base64_char_ratio,
                                      nonascii_ratio, _args_canonical,
                                      STEP_TYPES, THRESHOLDS, RNG_SEED, N_FOLDS)

VIEWS = {
    "full": ["obs_len", "obs_entropy", "obs_base64_ratio", "obs_nonascii_ratio",
             "thought_len", "thought_entropy", "arg_len", "has_tool_name",
             "memory_length"] + [f"step_type_{t}" for t in STEP_TYPES],
    "strict": ["obs_len", "obs_entropy", "obs_base64_ratio", "obs_nonascii_ratio",
               "thought_len", "thought_entropy", "arg_len", "has_tool_name"]
              + [f"step_type_{t}" for t in STEP_TYPES],
    "obs_only": ["obs_len", "obs_entropy", "obs_base64_ratio", "obs_nonascii_ratio"],
}


def step_features(step):
    obs = step.get("observation") or ""
    thought = step.get("thought") or ""
    tc = step.get("tool_call") or {}
    st = step.get("step_type")
    d = {
        "obs_len": float(len(obs)),
        "obs_entropy": shannon_entropy(obs),
        "obs_base64_ratio": base64_char_ratio(obs),
        "obs_nonascii_ratio": nonascii_ratio(obs),
        "thought_len": float(len(thought)),
        "thought_entropy": shannon_entropy(thought),
        "arg_len": float(len(_args_canonical(tc))),
        "has_tool_name": 1.0 if tc.get("name") else 0.0,
        "memory_length": float(step.get("memory_length", 0) or 0),
    }
    for t in STEP_TYPES:
        d[f"step_type_{t}"] = 1.0 if st == t else 0.0
    return d


def build_traces(runs_train, runs_test, names, label_mode):
    Xs, ys = [], []
    for r in runs_train:
        run_y = 1 if r["outcome"] == "unsafe" else 0
        for s in r["steps"]:
            fd = step_features(s)
            Xs.append([fd[n] for n in names])
            ys.append(run_y if label_mode == "run" else int(step_is_injected(s)))
    clf, scaler = fit_gb(np.array(Xs), np.array(ys))
    out = {}
    for r in runs_test:
        steps = r["steps"] or []
        if not steps:
            out[r["run_id"]] = {1: 0.0}
            continue
        X = np.array([[step_features(s)[n] for n in names] for s in steps])
        p = clf.predict_proba(scaler.transform(X))[:, 1]
        out[r["run_id"]] = {N: float(p[N - 1]) for N in range(1, len(steps) + 1)}
    return out, clf


def oof(runs, names, label_mode):
    y = np.array([1 if r["outcome"] == "unsafe" else 0 for r in runs])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    traces, imps = {}, []
    for tr_idx, te_idx in skf.split(np.zeros((len(runs), 1)), y):
        t, clf = build_traces([runs[i] for i in tr_idx],
                              [runs[i] for i in te_idx], names, label_mode)
        traces.update(t)
        imps.append(clf.feature_importances_)
    imp = dict(zip(names, np.mean(imps, axis=0).round(4).tolist()))
    return traces, imp


def main():
    out = {"meta": {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "views": VIEWS, "seed": RNG_SEED}}
    for regime, tag in (("a", "ciphertext"), ("b", "wordcodec_A")):
        print(f"\n=== {tag} ===")
        runs = load_regime(regime)
        block = {"detectors": {}, "importances": {}}
        traces_by_det = {}
        for view, names in VIEWS.items():
            for label_mode in ("run", "step"):
                key = f"perobs-{view}-{label_mode}"
                t0 = time.time()
                traces, imp = oof(runs, names, label_mode)
                traces_by_det[key] = traces
                block["detectors"][key] = evaluate(runs, traces, THRESHOLDS, key)
                block["importances"][key] = imp
                print(f"  {key:28s} {time.time()-t0:5.0f}s")
        block["matched_fpr"] = matched_fpr_table(runs, traces_by_det)
        out[tag] = block
        with open(os.path.join(HERE, "..", "results", "perobs_ablation.json"), "w") as f:
            json.dump(out, f, indent=1)
    print("\nDONE")


if __name__ == "__main__":
    main()

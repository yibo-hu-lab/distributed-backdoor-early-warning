import math
from collections import Counter
import numpy as np
import json

BASE64_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
)


ORIGINAL_FEATURES = [
    "n_steps_seen", "n_unique_agents", "n_tool_calls", "n_parse_errors",
    "n_final_answers", "frac_parse_errors", "avg_memory_length",
    "n_distinct_tools", "thought_avg_len", "obs_avg_len",
]


LENGTH_FEATURES = {"obs_avg_len", "thought_avg_len"}


NO_LENGTH_FEATURES = [f for f in ORIGINAL_FEATURES if f not in LENGTH_FEATURES]


GROUP_A_FEATURES = [
    "obs_entropy_mean", "obs_entropy_max", "obs_entropy_std",
    "obs_base64_char_ratio_mean", "obs_nonascii_ratio_mean",
    "obs_has_steg_marker_frac",
]


GROUP_B_FEATURES = ["thought_entropy_mean", "thought_uniqueness"]


GROUP_C_FEATURES = [
    "tool_arg_len_variance", "n_unique_arg_combinations", "arg_repetition_rate",
]


GROUP_D_FEATURES = [
    "duplicate_call_count", "duplicate_call_rate", "first_duplicate_position",
]


GROUP_E_FEATURES = ["per_agent_obs_entropy_spread"]


CONTENT_FEATURES = (
    NO_LENGTH_FEATURES + GROUP_A_FEATURES + GROUP_B_FEATURES +
    GROUP_C_FEATURES + GROUP_D_FEATURES + GROUP_E_FEATURES
)


STEG_MARKER = "||STEG||"


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def base64_char_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for ch in s if ch in BASE64_CHARS) / len(s)


def nonascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for ch in s if ord(ch) > 127) / len(s)


def _args_canonical(tc: dict) -> str:
    args = tc.get("args") or {}
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(args)


def compute_group_a(prefix):
    obs_list = [s.get("observation") or "" for s in prefix]
    if not obs_list:
        return {name: 0.0 for name in GROUP_A_FEATURES}
    ents = [shannon_entropy(o) for o in obs_list]
    return {
        "obs_entropy_mean": float(np.mean(ents)),
        "obs_entropy_max": float(np.max(ents)),
        "obs_entropy_std": float(np.std(ents)),
        "obs_base64_char_ratio_mean": float(np.mean([base64_char_ratio(o) for o in obs_list])),
        "obs_nonascii_ratio_mean": float(np.mean([nonascii_ratio(o) for o in obs_list])),
        "obs_has_steg_marker_frac": float(np.mean([STEG_MARKER in o for o in obs_list])),
    }


def compute_group_b(prefix):
    thoughts = [s.get("thought") or "" for s in prefix]
    n = len(thoughts)
    if n == 0:
        return {name: 0.0 for name in GROUP_B_FEATURES}
    return {
        "thought_entropy_mean": float(np.mean([shannon_entropy(t) for t in thoughts])),
        "thought_uniqueness": len(set(thoughts)) / n,
    }


def compute_group_c(prefix):
    tool_steps = [s for s in prefix if (s.get("tool_call") or {}).get("name")]
    if not tool_steps:
        return {name: 0.0 for name in GROUP_C_FEATURES}
    args_strs, combos = [], []
    for s in tool_steps:
        tc = s["tool_call"]
        args_str = _args_canonical(tc)
        args_strs.append(args_str)
        combos.append((tc.get("name"), args_str))
    arg_lens = [len(a) for a in args_strs]
    seen = set()
    repeats = 0
    for c in combos:
        if c in seen:
            repeats += 1
        else:
            seen.add(c)
    return {
        "tool_arg_len_variance": float(np.var(arg_lens)) if len(arg_lens) > 1 else 0.0,
        "n_unique_arg_combinations": float(len(set(combos))),
        "arg_repetition_rate": repeats / len(combos),
    }


def compute_group_d(prefix):
    n = len(prefix)
    if n == 0:
        return {name: 0.0 for name in GROUP_D_FEATURES}
    dup_idx = [i for i, s in enumerate(prefix) if s.get("step_type") == "duplicate_call"]
    dup_count = len(dup_idx)
    first_dup = (dup_idx[0] / n) if dup_idx else 1.0
    return {
        "duplicate_call_count": float(dup_count),
        "duplicate_call_rate": dup_count / n,
        "first_duplicate_position": float(first_dup),
    }


def compute_group_e(prefix):
    per_agent = {}
    for s in prefix:
        a = s.get("agent_id", "")
        per_agent.setdefault(a, []).append(shannon_entropy(s.get("observation") or ""))
    if len(per_agent) < 2:
        return {"per_agent_obs_entropy_spread": 0.0}
    means = [float(np.mean(ents)) for ents in per_agent.values() if ents]
    if not means:
        return {"per_agent_obs_entropy_spread": 0.0}
    return {"per_agent_obs_entropy_spread": float(max(means) - min(means))}


def extract_all_features(steps, max_n):
    prefix = steps[:max_n]
    n = len(prefix)
    features = {}

    if n == 0:
        features.update({name: 0 for name in ORIGINAL_FEATURES})
    else:
        n_tool_calls = sum(1 for s in prefix if s.get("step_type") == "tool_call")
        n_parse_errors = sum(1 for s in prefix if s.get("step_type") == "parse_error")
        n_final_answers = sum(1 for s in prefix if s.get("step_type") == "final_answer")

        agents, tools = set(), set()
        thought_lens, obs_lens, mem_lens = [], [], []
        for s in prefix:
            agents.add(s.get("agent_id", ""))
            tc = s.get("tool_call") or {}
            if tc.get("name"):
                tools.add(tc["name"])
            thought_lens.append(len(s.get("thought") or ""))
            obs_lens.append(len(s.get("observation") or ""))
            mem_lens.append(s.get("memory_length", 0) or 0)

        features.update({
            "n_steps_seen": n,
            "n_unique_agents": len(agents),
            "n_tool_calls": n_tool_calls,
            "n_parse_errors": n_parse_errors,
            "n_final_answers": n_final_answers,
            "frac_parse_errors": n_parse_errors / n if n else 0,
            "avg_memory_length": float(np.mean(mem_lens)) if mem_lens else 0,
            "n_distinct_tools": len(tools),
            "thought_avg_len": float(np.mean(thought_lens)) if thought_lens else 0,
            "obs_avg_len": float(np.mean(obs_lens)) if obs_lens else 0,
        })

    features.update(compute_group_a(prefix))
    features.update(compute_group_b(prefix))
    features.update(compute_group_c(prefix))
    features.update(compute_group_d(prefix))
    features.update(compute_group_e(prefix))
    return features


SPLIT_FEATURE_SET = [f for f in CONTENT_FEATURES if f != "obs_has_steg_marker_frac"]

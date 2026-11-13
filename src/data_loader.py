import argparse
import json
import os
from collections import defaultdict


STEP_FIELDS = [
    "step_index", "agent_id", "agent_role", "thought", "tool_call",
    "observation", "observation_clean", "step_type", "memory_length", "timestamp",
]

ATTACK_KEYWORDS = ["attack", "poison", "unsafe", "injected", "steg"]


def load_trajectories(data_dir, exclude_errors=True):
    jsonl_files = []
    for root, _dirs, files in os.walk(data_dir):
        for fname in sorted(files):
            if fname.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, fname))

    if not jsonl_files:
        print(f"WARNING: No .jsonl files found in {data_dir}")
        return []

    runs_by_id = defaultdict(lambda: {"rows": []})

    for fpath in jsonl_files:
        fname = os.path.basename(fpath)
        parent = os.path.basename(os.path.dirname(fpath))
        domain = parent if parent in ("api_tools", "database_searching") else "unknown"
        rel_parts = os.path.normpath(fpath).split(os.sep)
        is_heldout = "heldout" in rel_parts

        with open(fpath) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"WARNING: Bad JSON at {fname}:{line_num}, skipping")
                    continue

                run_id = row.get("run_id", f"unknown_{fname}_{line_num}")
                key = (fpath, run_id)
                entry = runs_by_id[key]
                entry["run_id"] = run_id
                entry["model"] = row.get("model", "unknown")
                entry["domain"] = domain
                entry["source_file"] = fname
                entry["source_path"] = fpath
                entry["is_heldout"] = is_heldout
                entry["task_id"] = row.get("task_id", entry.get("task_id"))
                entry["outcome"] = row.get("outcome", "unknown")
                entry["total_steps"] = row.get("total_steps", 0)
                entry["global_instruction"] = row.get("global_instruction", "")
                entry["rows"].append(row)

    runs = []
    for (_fpath, _run_id), entry in runs_by_id.items():
        outcome = entry["outcome"]
        if exclude_errors and outcome == "error":
            continue

        sorted_rows = sorted(entry["rows"], key=lambda r: r.get("step_index", 0))

        steps = []
        n_injected = 0
        for row in sorted_rows:
            step = {field: row.get(field, "" if field != "memory_length" else 0)
                    for field in STEP_FIELDS}
            if not isinstance(step["tool_call"], dict):
                step["tool_call"] = {"name": None, "args": {}}
            steps.append(step)

            obs = row.get("observation") or ""
            obs_clean = row.get("observation_clean") or ""
            if obs != obs_clean:
                n_injected += 1

        has_injection = n_injected > 0

        fname_lower = entry["source_file"].lower()
        is_from_attack_file = any(kw in fname_lower for kw in ATTACK_KEYWORDS)
        if not is_from_attack_file:
            is_from_attack_file = has_injection

        runs.append({
            "run_id": entry["run_id"],
            "model": entry["model"],
            "domain": entry["domain"],
            "source_file": entry["source_file"],
            "source_path": entry.get("source_path", ""),
            "is_heldout": entry.get("is_heldout", False),
            "task_id": entry.get("task_id"),
            "outcome": outcome,
            "steps": steps,
            "has_injection": has_injection,
            "n_injected_steps": n_injected,
            "is_from_attack_file": is_from_attack_file,
            "total_steps": entry["total_steps"],
            "global_instruction": entry["global_instruction"],
        })

    return runs


def print_summary(runs):
    print(f"Total runs: {len(runs)}")

    from collections import Counter
    outcomes = Counter(r["outcome"] for r in runs)
    print(f"Outcomes: {dict(outcomes)}")

    injected = sum(1 for r in runs if r["has_injection"])
    print(f"Runs with injection: {injected}")
    print(f"Clean runs: {len(runs) - injected}")

    models = Counter(r["model"] for r in runs)
    print(f"\nBy model:")
    for model, count in sorted(models.items()):
        inj = sum(1 for r in runs if r["model"] == model and r["has_injection"])
        print(f"  {model}: {count} runs ({inj} injected)")

    domains = Counter(r["domain"] for r in runs)
    print(f"\nBy domain:")
    for domain, count in sorted(domains.items()):
        inj = sum(1 for r in runs if r["domain"] == domain and r["has_injection"])
        print(f"  {domain}: {count} runs ({inj} injected)")

    files = Counter(r["source_file"] for r in runs)
    print(f"\nBy source file:")
    for fname, count in sorted(files.items()):
        inj = sum(1 for r in runs if r["source_file"] == fname and r["has_injection"])
        print(f"  {fname}: {count} runs ({inj} injected)")

    step_counts = [len(r["steps"]) for r in runs]
    print(f"\nSteps per run: min={min(step_counts)}, max={max(step_counts)}, "
          f"avg={sum(step_counts)/len(step_counts):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and summarize trajectory data")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__),
                        "..", "data", "trajectories"),
                        help="Root directory containing trajectory JSONL files")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    print(f"Loading trajectories from: {data_dir}")
    runs = load_trajectories(data_dir)
    print_summary(runs)

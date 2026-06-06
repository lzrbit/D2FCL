"""Parse training logs to compute per-round average time per algorithm."""
import os, re
from datetime import datetime
from collections import defaultdict
import json

ROOT = "results/paper_experiments"
TS_RE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Average accuracy')

def parse_log(path):
    """Use 'Average accuracy' lines as round-end anchors."""
    entries = []
    with open(path) as f:
        for line in f:
            m_ts = TS_RE.match(line)
            if m_ts:
                ts = datetime.strptime(m_ts.group(1), "%Y-%m-%d %H:%M:%S")
                entries.append(ts)
    return entries

def avg_round_time(path):
    entries = parse_log(path)
    if len(entries) < 2:
        return None, 0
    diffs = []
    for i in range(1, len(entries)):
        d = (entries[i] - entries[i - 1]).total_seconds()
        if 0 < d < 600:
            diffs.append(d)
    if not diffs:
        return None, 0
    return sum(diffs) / len(diffs), len(diffs)

results = {}
for category in ["baselines", "ablation"]:
    for ds in ["EMNIST-Letters", "CIFAR100"]:
        base = os.path.join(ROOT, category, ds)
        if not os.path.isdir(base):
            continue
        for sub in sorted(os.listdir(base)):
            sub_path = os.path.join(base, sub)
            if not os.path.isdir(sub_path):
                continue
            if category == "baselines":
                runs = [sub_path]
                algo = sub.split("_")[0]
            else:
                runs = [
                    os.path.join(sub_path, x) for x in os.listdir(sub_path)
                    if os.path.isdir(os.path.join(sub_path, x))
                ]
                algo = sub
            for run in runs:
                log = os.path.join(run, "training.log")
                if not os.path.isfile(log):
                    continue
                avg, n = avg_round_time(log)
                if avg is not None:
                    key = f"{ds}|{category}|{algo}"
                    if key not in results or results[key][1] < n:
                        results[key] = (avg, n)

print(f"{'Dataset':<16} {'Cat':<10} {'Algo':<30} {'AvgRound(s)':>12} {'N':>5}")
print("-" * 80)
for k in sorted(results.keys()):
    ds, cat, algo = k.split("|")
    avg, n = results[k]
    print(f"{ds:<16} {cat:<10} {algo:<30} {avg:>12.3f} {n:>5}")

with open("paper/figure_data/round_timings.json", "w") as f:
    json.dump({k: {"avg_round_sec": v[0], "n_rounds": v[1]} for k, v in results.items()}, f, indent=2)
print("\nSaved to paper/figure_data/round_timings.json")

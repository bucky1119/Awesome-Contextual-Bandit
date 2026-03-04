import numpy as np, json, os

print("=== 20 Newsgroups (2000r, seed=42) ===")
ng_dir = "results/newsgroups_2000r_seed42"
sp = os.path.join(ng_dir, "summary.json")
if os.path.exists(sp):
    with open(sp) as f:
        s = json.load(f)
    for k, v in sorted(s.items()):
        if k == "meta":
            continue
        print(f"  {k:20s}  regret={v['cumulative_regret']:8.1f}  reward={v['average_reward']:.4f}  acc={v['accuracy']:.4f}")

print()
print("=== Statlog results ===")
for d in sorted(os.listdir("results")):
    fp = os.path.join("results", d)
    if os.path.isdir(fp):
        print(f"  Dir: {d}/")
        for ff in sorted(os.listdir(fp)):
            print(f"    {ff}")

# Check latest JSON results
for f in sorted(os.listdir("results")):
    if f.endswith(".json"):
        fp = os.path.join("results", f)
        with open(fp) as fh:
            data = json.load(fh)
        print(f"\n=== {f} ===")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and "cumulative_regret" in v:
                    print(f"  {k:20s}  regret={v['cumulative_regret']:8.1f}")

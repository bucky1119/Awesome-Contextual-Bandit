import json
import os
import sys
import numpy as np

def analyze_jsonl(log_path, out_dir):
    if not os.path.exists(log_path):
        print(f"[ERROR] Log file not found: {log_path}")
        return
        
    history = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            history.append(json.loads(line))
            
    print(f"Loaded {len(history)} records from {log_path}")
    
    stages = {"<=1000(cold)": 1000, "<=2000": 2000, "<=3000": 3000, "All": len(history)}
    dominance = {}
    
    for stage_name, limit in stages.items():
        sub = history[:limit]
        cnt = {"s1": 0, "s2": 0, "s3": 0}
        
        rad_s1, rad_s2, rad_s3 = [], [], []
        mean_s1, mean_s2, mean_s3 = [], [], []
        
        for rec in sub:
            ms = rec.get("min_source", "unknown")
            if ms in cnt:
                cnt[ms] += 1
            
            # radius
            if rec.get("radius_linucb"):
                rad_s1.append(np.mean(rec["radius_linucb"]))
            if rec.get("radius_ucb1"):
                rad_s2.append(np.mean(rec["radius_ucb1"]))
            if rec.get("radius_llm"):
                rad_s3.append(np.mean(rec["radius_llm"]))
                
            # ucb means
            if rec.get("ucb_s1"):
                mean_s1.append(np.mean(rec["ucb_s1"]))
            if rec.get("ucb_s2"):
                mean_s2.append(np.mean(rec["ucb_s2"]))
            if rec.get("ucb_s3"):
                mean_s3.append(np.mean(rec["ucb_s3"]))
                
        total = sum(cnt.values())
        if total > 0:
            pct = {k: v/total for k, v in cnt.items()}
        else:
            pct = cnt
            
        dominance[stage_name] = {
            "counts": cnt,
            "percentages": pct,
            "radius_mean_s1": float(np.mean(rad_s1)) if rad_s1 else 0.0,
            "radius_mean_s2": float(np.mean(rad_s2)) if rad_s2 else 0.0,
            "radius_mean_s3": float(np.mean(rad_s3)) if rad_s3 else 0.0,
            "ucb_mean_s1": float(np.mean(mean_s1)) if mean_s1 else 0.0,
            "ucb_mean_s2": float(np.mean(mean_s2)) if mean_s2 else 0.0,
            "ucb_mean_s3": float(np.mean(mean_s3)) if mean_s3 else 0.0,
        }
        
    out_path = os.path.join(out_dir, "ourmethod_ucb_dominance_analysis_5000r.json")
    with open(out_path, "w") as f:
        json.dump(dominance, f, indent=2)
        
    print(f"Dominance analysis dumped to {out_path}")
    print(json.dumps(dominance, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        pass
    analyze_jsonl(sys.argv[1], sys.argv[2])

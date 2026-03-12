import numpy as np

def extract_ucb_stats(history):
    stages = {"<=200": 200, "<=500": 500, "All": len(history)}
    dominance = {}
    
    for stage_name, limit in stages.items():
        sub = history[:limit]
        cnt = {"s1": 0, "s2": 0, "s3": 0}
        rad_s1, rad_s2, rad_s3 = [], [], []
        ucb_s1, ucb_s2, ucb_s3 = [], [], []
        
        for rec in sub:
            ms = getattr(rec, "min_source", "unknown")
            if ms in cnt: cnt[ms] += 1
            if getattr(rec, "radius_linucb", None): rad_s1.append(np.mean(rec.radius_linucb))
            if getattr(rec, "radius_ucb1", None): rad_s2.append(np.mean(rec.radius_ucb1))
            if getattr(rec, "radius_llm", None): rad_s3.append(np.mean(rec.radius_llm))
            
            if getattr(rec, "ucb_s1", None): ucb_s1.append(np.mean(rec.ucb_s1))
            if getattr(rec, "ucb_s2", None): ucb_s2.append(np.mean(rec.ucb_s2))
            if getattr(rec, "ucb_s3", None): ucb_s3.append(np.mean(rec.ucb_s3))
            
        total = sum(cnt.values())
        dominance[stage_name] = {
            "counts": cnt,
            "percentages": {k: v/total for k, v in cnt.items()} if total else cnt,
            "radius_mean_s1": float(np.mean(rad_s1)) if rad_s1 else 0.0,
            "radius_mean_s2": float(np.mean(rad_s2)) if rad_s2 else 0.0,
            "radius_mean_s3": float(np.mean(rad_s3)) if rad_s3 else 0.0,
            "ucb_mean_s1": float(np.mean(ucb_s1)) if ucb_s1 else 0.0,
            "ucb_mean_s2": float(np.mean(ucb_s2)) if ucb_s2 else 0.0,
            "ucb_mean_s3": float(np.mean(ucb_s3)) if ucb_s3 else 0.0,
        }
    return dominance

import os
import sys
import numpy as np
import json
from collections import defaultdict
import time
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run_dataset_step import load_ag_news
from OurMethod.run_full_experiment import run_our_method_pipeline
from OurMethod.core.protocol import Arm
from OurMethod.core.prompt_builder import StructuredPromptBuilder
from OurMethod.core.cold_start_cache import ColdStartCache

def build_ag_news_arms_and_pb():
    categories = [
        ("World", "international politics and global events"),
        ("Sports", "sports competitions and athletes"),
        ("Business", "economy, finance, companies"),
        ("Sci/Tech", "science and technology innovations"),
    ]
    arms = [Arm(arm_id=i, name=n, description=desc) for i, (n, desc) in enumerate(categories)]
    pb = StructuredPromptBuilder(
        role="news topic classification expert",
        scenario=(
            "AG News topic classification as contextual bandit. "
            "Given the text of a news article, predict which topic it belongs to."
        ),
        max_feedback=5,
        max_feature_display=0,
    )
    return arms, pb

def extract_and_save_ucb_stats(history, dataset_name, out_dir):cmab, opt_rewards, opt_actions, num_actions, context_dim, n_rounds, 
              cold_start_n, model_name, dump_prompts=0):
    arms, pb = build_ag_news_arms_and_pb()
    
    res = run_our_method_pipeline(
        cmab=cmab,
        opt_rewards=opt_rewards,
        opt_actions=opt_actions,
        n_rounds=n_rounds,
        num_actions=num_actions,
        context_dim=context_dim,
        cold_start_n=cold_start_n,
        model_name=model_name,
        seed=42,
        z_dim=16,
        dataset_name="ag_news",
        arms=arms,
        prompt_builder=pb,
        dump_prompts=dump_prompts,
        no_cache=False,  # Use cache when possible!
    )
    return res

def analyze_history(history, dataset_name="ag_news"):
    out_dir = os.path.join(ROOT, "results", "ourmethod_debug", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    
    stages = {"<=200": 200, "<=500": 500, "All": len(history)}
    dominance = {}
    
    for stage_name, limit in stages.items():
        sub = history[:limit]
        cnt = {"s1": 0, "s2": 0, "s3": 0}
        
        rad_s1, rad_s2, rad_s3 = [], [], []
        mean_s1, mean_s2, mean_s3 = [], [], []
        
        for rec in sub:
            ms = getattr(rec, "min_source", "unknown")
            if ms in cnt:
                cnt[ms] += 1
            
            if hasattr(rec, "radius_linucb") and rec.radius_linucb:
                rad_s1.append(np.mean(rec.radius_linucb))
            if hasattr(rec, "radius_ucb1") and rec.radius_ucb1:
                rad_s2.append(np.mean(rec.radius_ucb1))
            if hasattr(rec, "radius_llm") and rec.radius_llm:
                rad_s3.append(np.mean(rec.radius_llm))
                
            if hasattr(rec, "ucb_s1") and rec.ucb_s1:
                mean_s1.append(np.mean(rec.ucb_s1))
            if hasattr(rec, "ucb_s2") and rec.ucb_s2:
                mean_s2.append(np.mean(rec.ucb_s2))
            if hasattr(rec, "ucb_s3") and rec.ucb_s3:
                mean_s3.append(np.mean(rec.ucb_s3))
        
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
    
    with open(os.path.join(out_dir, "ucb_dominance.json"), "w") as f:
        json.dump(dominance, f, indent=2)
    return dominance

def analyze_cold_start_cache(model_name, cold_start_n, seed=42, dataset_name="ag_news"):
    from OurMethod.core.cold_start_cache import ColdStartCache
    cache = ColdStartCache()
    tag = cache.make_tag(dataset_name, model_name, cold_start_n, seed)
    
    out_dir = os.path.join(ROOT, "results", "ourmethod_debug", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    
    stats = {
        "parse_success_rate": 0.0,
        "missing_arms_count": 0,
        "fallback_count": 0,
        "variance_mean": 0.0
    }
    
    if not cache.exists(tag):
        print(f"[WARN] Cache not found for {tag}")
        return stats
        
    data = cache.load(tag)
        
    all_ds = data.get("all_ds", [])
    if not all_ds:
        return stats
        
    valid_dsCount = 0
    vars_list = []
    
    for ds_list in all_ds:
        if ds_list is None or len(ds_list) != 4:
            continue
        valid_dsCount += 1
        
        means = []
        for ds in ds_list:
            means.append(ds.predicted_mean)
            if "(missing from LLM output)" in ds.reasoning:
                stats["missing_arms_count"] += 1
            if "(fallback" in ds.reasoning:
                stats["fallback_count"] += 1
        vars_list.append(np.var(means))
        
    stats["parse_success_rate"] = valid_dsCount / len(all_ds)
    stats["variance_mean"] = float(np.mean(vars_list)) if vars_list else 0.0
    
    with open(os.path.join(out_dir, "ds_llm_quality.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="stub")
    args = parser.parse_args()
    
    print("Loading AG News dataset...")
    cmab, opt_rewards, opt_actions, num_actions, context_dim, _ = load_ag_news(2000, 200, 42)
    
    out_dir = os.path.join(ROOT, "results", "ourmethod_debug", "ag_news")
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n--- Running Group A (cs=50) ---")
    res_A = run_group(cmab, opt_rewards, opt_actions, num_actions, context_dim, 
                      n_rounds=2000, cold_start_n=50, model_name=args.model, dump_prompts=3)
    
    print("\n--- Running Group B (cs=200) ---")
    res_B = run_group(cmab, opt_rewards, opt_actions, num_actions, context_dim, 
                      n_rounds=2000, cold_start_n=200, model_name=args.model, dump_prompts=0)
                      
    print("\nAnalyzing ds_llm quality from group B cache...")
    ds_stats = analyze_cold_start_cache(args.model, 200, 42, "ag_news")
    print("ds_llm_quality:", json.dumps(ds_stats, indent=2))
    
    print("\nAnalyzing UCB dominance from group B history...")
    dom_stats = analyze_history(res_B["history"], "ag_news")
    print("ucb_dominance (Group B):", json.dumps(dom_stats, indent=2))
    
    ablation = {
        "Group_A_cs50": {
            "avg_reward": float(np.mean(res_A["rewards"])),
            "cum_regret": float(res_A["cumulative_regret"][-1])
        },
        "Group_B_cs200": {
            "avg_reward": float(np.mean(res_B["rewards"])),
            "cum_regret": float(res_B["cumulative_regret"][-1])
        },
        "Group_C_cs200_no_gen_online": {
            "avg_reward": float(np.mean(res_B["rewards"])), 
            "cum_regret": float(res_B["cumulative_regret"][-1]),
            "note": "Online is already NO_GEN by default in OurMethod."
        }
    }
    
    with open(os.path.join(out_dir, "ablation_summary.json"), "w") as f:
        json.dump(ablation, f, indent=2)
        
    print("\n=== FINAL RESULTS SUMMARY ===")
    print("Ablation:", json.dumps(ablation, indent=2))
        
if __name__ == "__main__":
    main()

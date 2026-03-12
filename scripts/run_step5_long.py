import os
import sys
import numpy as np
import json
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run_dataset_step import load_ag_news
from OurMethod.run_full_experiment import run_our_method_pipeline
from scripts.verify_ag_news import build_ag_news_arms_and_pb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2_5_7b")
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--cold_start", type=int, default=200)
    args = parser.parse_args()
    
    print(f"Loading AG News dataset for {args.rounds} rounds...")
    cmab, opt_rewards, opt_actions, num_actions, context_dim, _ = load_ag_news(args.rounds, args.cold_start, 42)
    
    arms, pb = build_ag_news_arms_and_pb()
    
    print(f"\n--- Running Step 5: Long Run ({args.rounds} rounds, cs={args.cold_start}) ---")
    res = run_our_method_pipeline(
        cmab=cmab,
        opt_rewards=opt_rewards,
        opt_actions=opt_actions,
        n_rounds=args.rounds,
        num_actions=num_actions,
        context_dim=context_dim,
        cold_start_n=args.cold_start,
        model_name=args.model,
        seed=42,
        z_dim=16,
        dataset_name="ag_news",
        arms=arms,
        prompt_builder=pb,
        dump_prompts=0,
        no_cache=False,
    )
    
    out_dir = os.path.join(ROOT, "results", "ourmethod_debug", "ag_news")
    os.makedirs(out_dir, exist_ok=True)
    
    summary = {
        "model": args.model,
        "n_rounds": args.rounds,
        "cold_start_n": args.cold_start,
        "final_avg_reward": float(np.mean(res["rewards"])),
        "final_cum_regret": float(res["cumulative_regret"][-1]),
        "periodic_reward_trends": []
    }
    
    # Calculate average reward in chunks of 500
    chunk_size = 500
    for i in range(0, args.rounds, chunk_size):
        chunk = res["rewards"][i:i+chunk_size]
        if len(chunk) > 0:
            summary["periodic_reward_trends"].append(float(np.mean(chunk)))
            
    with open(os.path.join(out_dir, "longrun_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n=== LONGRUN SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

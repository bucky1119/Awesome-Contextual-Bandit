import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from run_dataset_step import load_ag_news
from OurMethod.run_full_experiment import run_our_method_pipeline
from scripts.verify_ag_news import build_ag_news_arms_and_pb

cmab, opt_rewards, opt_actions, num_actions, context_dim, _ = load_ag_news(20, 20, 42)
arms, pb = build_ag_news_arms_and_pb()

res = run_our_method_pipeline(
    cmab=cmab, opt_rewards=opt_rewards, opt_actions=opt_actions,
    n_rounds=20, num_actions=num_actions, context_dim=context_dim,
    cold_start_n=20, model_name="stub", seed=42, z_dim=16,
    dataset_name="ag_news", arms=arms, prompt_builder=pb,
    dump_prompts=0, no_cache=False
)

rec = res["history"][5]
print("UCB VALUES:", rec.ucb_values)
print("MIN SOURCE:", rec.min_source)
print("ALGORITHM:", rec.algorithm)

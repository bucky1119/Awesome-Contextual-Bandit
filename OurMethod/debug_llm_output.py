#!/usr/bin/env python3
"""Quick diagnostic: check what SmolLM2 generates for one context."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from OurMethod.core.frozen_llm import GenerativeFrozenLLM
from OurMethod.core.prompt_builder import StructuredPromptBuilder
from OurMethod.core.protocol import Arm, Context
from OurMethod.run_statlog import StatlogBanditEnv

env = StatlogBanditEnv("datasets/statlog.trn", num_contexts=5, seed=42)
arms = [Arm(arm_id=i, name=f"class_{i+1}") for i in range(7)]
pb = StructuredPromptBuilder(max_feedback=3)
llm = GenerativeFrozenLLM(
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    max_new_tokens=384,
    temperature=0.1,
)

ctx = Context(features=env.contexts[0])
prompt = pb.build(ctx, arms)
print("=== PROMPT (first 600 chars) ===")
print(prompt[:600])
print("...")
print()

h_t, ds_llm = llm.encode(prompt, num_arms=7)
print(f"h_t shape: {h_t.shape}, norm: {np.linalg.norm(h_t):.4f}")
print()
print("=== ds_llm results ===")
if ds_llm:
    for d in ds_llm:
        print(f"  arm {d.arm_id}: mean={d.predicted_mean:.4f}, "
              f"std={d.predicted_std:.4f}, reason={d.reasoning[:80]}")
    print()
    print("=== LLM raw output (first 500 chars) ===")
    print(ds_llm[0].raw_text[:500])
else:
    print("ds_llm is None")

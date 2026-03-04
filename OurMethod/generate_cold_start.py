#!/usr/bin/env python3
"""冷启动仿真数据预生成脚本。

对指定数据集+模型组合，一次性运行 LLM 冷启动并缓存结果。
之后的实验可直接加载缓存，跳过耗时的 LLM 生成过程。

用法:
  # 生成 SmolLM2 对 Statlog 的冷启动缓存 (50 contexts)
  python OurMethod/generate_cold_start.py --model smollm2 --dataset statlog \\
      --cold_start_n 50 --seed 42

  # 生成 stub 模型缓存 (快速测试)
  python OurMethod/generate_cold_start.py --model stub --dataset statlog \\
      --cold_start_n 50 --seed 42

  # 查看已有缓存
  python OurMethod/generate_cold_start.py --list

  # 强制重新生成 (覆盖已有缓存)
  python OurMethod/generate_cold_start.py --model smollm2 --dataset statlog \\
      --cold_start_n 50 --seed 42 --force
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from OurMethod.core.protocol import Arm, Context
from OurMethod.core.prompt_builder import StructuredPromptBuilder
from OurMethod.core.frozen_llm import (
    StubFrozenLLM, GenerativeFrozenLLM, FrozenLLMRegistry, _HAS_TRANSFORMERS,
)
from OurMethod.core.combined_policy import CombinedUCBPolicy
from OurMethod.core.cold_start import ColdStartSimulator
from OurMethod.core.cold_start_cache import ColdStartCache

from bandits.core.contextual_bandit import ContextualBandit
from bandits.data.data_sampler import sample_statlog_data


# ====================================================================== #
#  加载数据集                                                             #
# ====================================================================== #

def load_dataset(dataset_name: str, n_contexts: int, seed: int):
    """加载数据集并返回 (cmab, num_actions, context_dim, opt_rewards, opt_actions)。"""
    np.random.seed(seed)

    if dataset_name == "statlog":
        data_path = os.path.join(ROOT, "datasets", "statlog.trn")
        if not os.path.exists(data_path):
            print(f"[ERROR] 找不到: {data_path}")
            sys.exit(1)
        n = min(n_contexts, 43500)
        dataset, (opt_rewards, opt_actions) = sample_statlog_data(
            data_path, n, shuffle_rows=True,
        )
        num_actions = 7
        context_dim = dataset.shape[1] - num_actions
        cmab = ContextualBandit(num_actions, context_dim)
        cmab.feed_data(dataset)
        return cmab, num_actions, context_dim
    else:
        print(f"[ERROR] 未知数据集: {dataset_name}")
        print(f"  支持: statlog")
        sys.exit(1)


# ====================================================================== #
#  构建模型                                                               #
# ====================================================================== #

def build_registry(model_name: str) -> FrozenLLMRegistry:
    reg = FrozenLLMRegistry()
    reg.register("stub", StubFrozenLLM(hidden_dim=128, generate_ds_llm=True))

    if _HAS_TRANSFORMERS and model_name == "smollm2":
        try:
            print("[INFO] 加载 SmolLM2-360M-Instruct ...")
            reg.register("smollm2", GenerativeFrozenLLM(
                model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
                max_new_tokens=384, temperature=0.1,
                use_chat_template=True, generate_ds_llm=True,
            ), default=True)
        except Exception as e:
            print(f"[WARN] 无法加载 SmolLM2: {e}")
    return reg


# ====================================================================== #
#  主流程                                                                 #
# ====================================================================== #

def generate(
    dataset_name: str,
    model_name: str,
    cold_start_n: int,
    seed: int,
    force: bool = False,
):
    """生成冷启动仿真数据并保存到缓存。"""

    cache = ColdStartCache()
    tag = cache.make_tag(dataset_name, model_name, cold_start_n, seed)

    if cache.exists(tag) and not force:
        print(f"[SKIP] 缓存已存在: {tag}")
        print(f"  路径: {cache.tag_dir(tag)}")
        print(f"  使用 --force 可强制重新生成")
        return tag

    t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"  冷启动数据生成")
    print(f"  dataset={dataset_name}, model={model_name}")
    print(f"  cold_start_n={cold_start_n}, seed={seed}")
    print(f"  tag={tag}")
    print(f"{'=' * 60}\n")

    # 1) 加载数据集
    cmab, num_actions, context_dim = load_dataset(
        dataset_name, cold_start_n, seed,
    )
    cs_contexts = [cmab.context(i) for i in range(cold_start_n)]
    print(f"  数据集: {dataset_name}, {context_dim}D features, {num_actions} actions")
    print(f"  采样 {cold_start_n} 个冷启动上下文")

    # 2) 构建 LLM + 组件
    reg = build_registry(model_name)
    if model_name not in reg:
        print(f"[ERROR] model '{model_name}' 不可用: {reg.list_models()}")
        sys.exit(1)
    llm = reg.get(model_name)
    h_dim = llm.get_hidden_dim()
    print(f"  LLM: {model_name} (h_dim={h_dim})")

    arms = [Arm(arm_id=i, name=f"class_{i + 1}",
                description=f"Class {i + 1}")
            for i in range(num_actions)]
    pb = StructuredPromptBuilder(
        role="classification expert",
        scenario=f"{dataset_name} classification as contextual bandit",
        max_feedback=5,
    )

    # 3) 创建临时 Policy 用于预填充
    policy = CombinedUCBPolicy(num_actions=num_actions, z_dim=16)

    # 4) 确保 LLM 生成 ds_llm
    prev_gen = llm.generate_ds_llm
    llm.generate_ds_llm = True

    # 5) 运行仿真
    simulator = ColdStartSimulator(llm, pb, arms)
    all_ds, all_h = simulator.warmup_policy(policy, cs_contexts, verbose=True)

    llm.generate_ds_llm = prev_gen

    # 6) 保存缓存
    metadata = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "context_dim": context_dim,
        "generation_time_sec": round(time.time() - t0, 1),
    }
    cache_dir = cache.save(
        tag=tag,
        all_h=all_h,
        all_ds=all_ds,
        sim_stats=policy.sim_stats,
        contexts=cs_contexts,
        metadata=metadata,
    )

    elapsed = time.time() - t0
    print(f"\n  缓存已保存: {cache_dir}")
    print(f"  SimStats: {policy.sim_stats.summary()}")
    print(f"  总耗时: {elapsed:.1f}s")
    return tag


def list_caches():
    """列出所有已有缓存。"""
    cache = ColdStartCache()
    tags = cache.list_tags()
    if not tags:
        print("无冷启动缓存。")
        return

    print(f"\n已有缓存 ({len(tags)} 个):")
    print(f"{'Tag':<45s} {'N':>5s} {'H_dim':>6s} {'Created':>22s}")
    print("-" * 80)
    for tag in tags:
        try:
            data = cache.load(tag)
            meta = data["metadata"]
            print(f"{tag:<45s} {meta.get('cold_start_n','?'):>5} "
                  f"{meta.get('h_dim','?'):>6} "
                  f"{meta.get('created_at','?'):>22s}")
        except Exception as e:
            print(f"{tag:<45s} [ERROR: {e}]")


def main():
    parser = argparse.ArgumentParser(
        description="冷启动仿真数据预生成：一次生成，多次复用")
    parser.add_argument("--dataset", type=str, default="statlog",
                        help="数据集名称: statlog")
    parser.add_argument("--model", type=str, default="stub",
                        help="LLM 模型: stub | smollm2")
    parser.add_argument("--cold_start_n", type=int, default=50,
                        help="冷启动上下文数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="强制重新生成 (覆盖已有缓存)")
    parser.add_argument("--list", action="store_true",
                        help="列出所有已有缓存")
    args = parser.parse_args()

    if args.list:
        list_caches()
        return

    generate(
        dataset_name=args.dataset,
        model_name=args.model,
        cold_start_n=args.cold_start_n,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()

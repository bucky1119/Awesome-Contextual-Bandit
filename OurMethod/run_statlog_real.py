#!/usr/bin/env python3
"""在 Statlog 数据集上用真实冻结 LLM 运行 Combined UCB 实验。

特性：
  ✓ 支持多种冻结 LLM（--model 选择）
  ✓ 冷启动阶段：LLM 预生成 ds_llm → 预填充 UCB^s3
  ✓ 在线阶段：提取 h_t → 压缩 → Combined UCB 决策
  ✓ 可选在线继续生成 ds_llm（--gen_online, 慢但更准）

预注册模型:
  stub         — 伪特征（基线对比）
  smollm2      — HuggingFaceTB/SmolLM2-360M-Instruct
    qwen2_5_7b   — Qwen/Qwen2.5-7B-Instruct
    llama3_1_8b  — meta-llama/Llama-3.1-8B-Instruct
  (可通过 --custom_model 加载任意 HuggingFace 模型)

用法:
  python OurMethod/run_statlog_real.py --model stub --n_rounds 2000
  python OurMethod/run_statlog_real.py --model smollm2 --n_rounds 2000 --cold_start_n 200
  python OurMethod/run_statlog_real.py --custom_model Qwen/Qwen2.5-0.5B-Instruct --n_rounds 1000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from OurMethod.core.protocol import Arm, Context, DecisionRecord
from OurMethod.core.prompt_builder import StructuredPromptBuilder
from OurMethod.core.frozen_llm import (
    StubFrozenLLM, GenerativeFrozenLLM, FrozenLLMRegistry,
    _HAS_TRANSFORMERS,
)
from OurMethod.core.compressor import MLPCompressor
from OurMethod.core.combined_policy import CombinedUCBPolicy
from OurMethod.core.online_runner import OnlineRunner
from OurMethod.core.cold_start import ColdStartSimulator
from OurMethod.run_statlog import StatlogBanditEnv


# ====================================================================== #
#  模型注册表                                                              #
# ====================================================================== #

def build_registry(custom_model: Optional[str] = None,
                   selected_model: str = "stub") -> FrozenLLMRegistry:
    """构建冻结 LLM 注册表（惰性加载：仅加载被选中的模型）。"""
    reg = FrozenLLMRegistry()

    # 总是注册 stub（无依赖，零开销）
    reg.register("stub", StubFrozenLLM(hidden_dim=128, generate_ds_llm=True))

    preset_models = {
        "smollm2": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "qwen2_5_7b": "/home/csg/Awesome-contextual-bandits/models/huggingface/Qwen2.5-7B-Instruct",
        "llama3_1_8b": "meta-llama/Llama-3.1-8B-Instruct",
    }

    # 仅当选中对应模型 且 transformers 可用 时才实际加载
    if _HAS_TRANSFORMERS and selected_model in preset_models:
        model_id = preset_models[selected_model]
        try:
            print(f"[INFO] 加载 {model_id} ...")
            reg.register(
                selected_model,
                GenerativeFrozenLLM(
                    model_name=model_id,
                    max_new_tokens=384,
                    temperature=0.1,
                    use_chat_template=True,
                    generate_ds_llm=True,
                ),
                default=True,
            )
        except Exception as e:
            print(f"[WARN] 无法加载 {model_id}: {e}")

    # 自定义模型
    if custom_model and _HAS_TRANSFORMERS:
        name = custom_model.split("/")[-1].lower().replace("-", "_")
        if selected_model == name or selected_model == custom_model:
            try:
                print(f"[INFO] 加载自定义模型: {custom_model} ...")
                reg.register(
                    name,
                    GenerativeFrozenLLM(
                        model_name=custom_model,
                        max_new_tokens=384,
                        temperature=0.1,
                        use_chat_template=True,
                        generate_ds_llm=True,
                    ),
                    default=True,
                )
                print(f"[INFO] 已注册 '{name}': {custom_model}")
            except Exception as e:
                print(f"[ERROR] 无法加载 {custom_model}: {e}")

    return reg


# ====================================================================== #
#  主实验                                                                  #
# ====================================================================== #

def run_experiment(
    model_name: str = "stub",
    custom_model: Optional[str] = None,
    n_rounds: int = 2000,
    cold_start_n: int = 200,
    gen_online: bool = False,
    seed: int = 42,
    z_dim: int = 16,
    c_ucb1: float = 1.0,
    alpha_linucb: float = 1.0,
    c_sim: float = 1.0,
    verbose: bool = True,
):
    """运行 Statlog 实验。

    Args:
        model_name:    注册表中的模型名 (stub / smollm2 / qwen2_5_7b / llama3_1_8b / ...)
        custom_model:  自定义 HuggingFace 模型路径（优先于 model_name）
        n_rounds:      在线运行轮次
        cold_start_n:  冷启动仿真上下文数量 (0=跳过)
        gen_online:    在线阶段是否继续生成 ds_llm (慢但更新 UCB^s3)
        seed:          随机种子
        z_dim:         压缩特征维度
        verbose:       是否打印详细信息
    """
    # ---- 1) 构建注册表并选择模型 ---- #
    effective_name = model_name
    if custom_model:
        effective_name = custom_model.split("/")[-1].lower().replace("-", "_")

    reg = build_registry(custom_model=custom_model, selected_model=effective_name)

    if effective_name not in reg:
        print(f"[ERROR] 模型 '{effective_name}' 不可用。可用: {reg.list_models()}")
        return None

    llm = reg.get(effective_name)
    h_dim = llm.get_hidden_dim()
    print(f"选定模型: {effective_name}  (hidden_dim={h_dim})")
    print(f"已注册模型: {reg.list_models()}")

    # ---- 2) 加载数据 ---- #
    data_path = os.path.join(ROOT, "datasets", "statlog.trn")
    if not os.path.exists(data_path):
        print(f"[ERROR] 找不到数据文件: {data_path}")
        return None

    env = StatlogBanditEnv(data_path, num_contexts=n_rounds + cold_start_n,
                           shuffle=True, seed=seed, normalize=True)
    K = env.num_actions
    d = env.d
    print(f"Statlog: {env.n} 行, {d} 维特征, {K} 个动作")

    # ---- 3) 构建组件 ---- #
    arms = [Arm(arm_id=i, name=f"class_{i+1}",
                description=f"Shuttle class {i+1}")
            for i in range(K)]

    pb = StructuredPromptBuilder(
        role="shuttle classification expert",
        scenario="Statlog shuttle type classification as contextual bandit",
        max_feedback=5,
    )

    comp = MLPCompressor(input_dim=h_dim, output_dim=z_dim)
    policy = CombinedUCBPolicy(
        num_actions=K, z_dim=z_dim,
        c_ucb1=c_ucb1, alpha_linucb=alpha_linucb, c_sim=c_sim,
    )

    # ---- 4) 冷启动 ---- #
    if cold_start_n > 0:
        print(f"\n--- 冷启动阶段 ({cold_start_n} contexts) ---")
        # 确保冷启动时 LLM 生成 ds_llm
        prev_gen = llm.generate_ds_llm
        llm.generate_ds_llm = True

        # 从数据集采样冷启动上下文
        cs_contexts = [env.contexts[i] for i in range(min(cold_start_n, env.n))]
        simulator = ColdStartSimulator(llm, pb, arms)
        simulator.warmup_policy(policy, cs_contexts, verbose=verbose)

        llm.generate_ds_llm = prev_gen
        print(f"SimStats after warmup: {policy.sim_stats.summary()}")
    else:
        print("\n--- 跳过冷启动 ---")

    # ---- 5) 设置在线阶段 LLM 生成行为 ---- #
    if not gen_online:
        llm.generate_ds_llm = False
        print("在线阶段: 仅提取 h_t (不生成 ds_llm，速度快)")
    else:
        llm.generate_ds_llm = True
        print("在线阶段: 提取 h_t + 生成 ds_llm (速度慢)")

    # ---- 6) 运行在线实验 ---- #
    print(f"\n--- 在线阶段 ({n_rounds} rounds) ---")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"statlog_{effective_name}_seed{seed}.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    # 重置环境游标（冷启动用了一部分）
    env._cursor = cold_start_n - 1

    runner = OnlineRunner(
        prompt_builder=pb,
        llm_encoder=llm,
        compressor=comp,
        policy=policy,
        arms=arms,
        env_reward_fn=env.reward,
        env_optimal_fn=env.optimal,
        context_sampler=env.sample_context,
        context_dim=d,
        log_path=log_path,
        seed=seed,
    )

    t0 = time.time()
    history = runner.run(n_rounds, verbose=verbose)
    elapsed = time.time() - t0

    # ---- 7) 汇总 ---- #
    rewards = [r.reward for r in history]
    regrets = [r.regret for r in history if r.regret is not None]
    accuracy = np.mean([1.0 if r.reward == 1.0 else 0.0 for r in history])
    cum_regret = sum(regrets)

    print(f"\n{'='*60}")
    print(f"Statlog 实验结果 — 模型: {effective_name}")
    print(f"{'='*60}")
    print(f"  耗时:          {elapsed:.1f}s ({elapsed/n_rounds*1000:.0f}ms/step)")
    print(f"  总轮次:        {n_rounds}")
    print(f"  冷启动:        {cold_start_n} contexts")
    print(f"  在线生成:      {'是' if gen_online else '否'}")
    print(f"  累计奖励:      {sum(rewards):.1f} / {n_rounds}")
    print(f"  平均奖励:      {np.mean(rewards):.4f}")
    print(f"  准确率:        {accuracy:.4f}")
    print(f"  累计遗憾:      {cum_regret:.1f}")

    # 分段统计
    seg_size = max(n_rounds // 5, 1)
    print(f"\n  分段统计:")
    for seg in range(5):
        lo = seg * seg_size
        hi = min(lo + seg_size, n_rounds)
        seg_r = rewards[lo:hi]
        seg_reg = regrets[lo:hi] if regrets else []
        seg_acc = np.mean([1.0 if r == 1.0 else 0.0 for r in seg_r])
        seg_avg = np.mean(seg_r) if seg_r else 0
        print(f"    [{lo:>5d}-{hi:>5d}]  avg={seg_avg:.4f}  "
              f"acc={seg_acc:.4f}  regret={sum(seg_reg):.1f}")

    print(f"\n  SimStats: {policy.sim_stats.summary()}")
    print(f"  日志: {log_path}")

    return history


def main():
    p = argparse.ArgumentParser(
        description="OurMethod on Statlog with Real Frozen LLM",
    )
    p.add_argument("--model", dest="model_name", type=str, default="stub",
                   help="模型名 (stub/smollm2/qwen2_5_7b/llama3_1_8b 或自定义)")
    p.add_argument("--custom_model", type=str, default=None,
                   help="自定义 HuggingFace 模型路径")
    p.add_argument("--n_rounds", type=int, default=2000)
    p.add_argument("--cold_start_n", type=int, default=200,
                   help="冷启动仿真数量 (0=跳过)")
    p.add_argument("--gen_online", action="store_true", default=False,
                   help="在线阶段继续生成 ds_llm")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--z_dim", type=int, default=16)
    p.add_argument("--c_ucb1", type=float, default=1.0)
    p.add_argument("--alpha_linucb", type=float, default=1.0)
    p.add_argument("--c_sim", type=float, default=1.0)
    p.add_argument("--verbose", action="store_true", default=True)
    args = p.parse_args()
    run_experiment(**vars(args))


if __name__ == "__main__":
    main()

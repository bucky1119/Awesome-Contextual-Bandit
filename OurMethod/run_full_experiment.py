#!/usr/bin/env python3
"""完整闭环实验：冷启动 → 热启动训练 → 在线决策 → 周期性离线训练 → 与 baseline 对比绘图。

完整流程:
  Phase 0: 冷启动 — LLM 生成仿真数据, 预填充 UCB^s3, 收集 (h_t, ds_llm) 构建仿真数据集
  Phase 1: 热启动训练 — 用仿真数据集训练 Compressor fφ (θ 为随机初始化, detach)
  Phase 2: 在线决策 — Combined UCB 选 arm，积累真实数据集 D
  Phase 3: 周期性离线训练 — 每 offline_freq 轮用 D 训练 fφ (θ 冻结于最新在线值)，重建 LinUCB
  Phase 4: 运行 baseline (neural_bandit, neural_linear) 对比
  Phase 5: 保存结果 + 绘制遗憾增长曲线

公平性保证:
  • 数据采样 total = cold_start_n + n_rounds
  • OurMethod 冷启动: 行 [0, cold_start_n) — 仅 LLM 仿真，不消耗真实奖励
  • 所有算法在线决策: 行 [cold_start_n, cold_start_n+n_rounds) — 同一组上下文

用法:
  python OurMethod/run_full_experiment.py --model smollm2 --n_rounds 5000 --cold_start_n 50
  python OurMethod/run_full_experiment.py --model stub --n_rounds 5000 --cold_start_n 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- OurMethod ---- #
from OurMethod.core.protocol import Arm, Context, DecisionRecord, DsLlm
from OurMethod.core.prompt_builder import StructuredPromptBuilder
from OurMethod.core.frozen_llm import (
    StubFrozenLLM, GenerativeFrozenLLM, MLXGenerativeFrozenLLM,
    FrozenLLMRegistry, _HAS_TRANSFORMERS, _HAS_MLX,
)
from OurMethod.core.compressor import MLPCompressor
from OurMethod.core.combined_policy import CombinedUCBPolicy
from OurMethod.core.cold_start import ColdStartSimulator
from OurMethod.core.offline_dataset import InMemoryOfflineDataset
from OurMethod.core.train_compressor import OfflineTrainer
from OurMethod.core.cold_start_cache import ColdStartCache
from OurMethod.core.checkpoint import CheckpointManager

# ---- baseline ---- #
from bandits.algorithms.neural_bandit_model import NeuralBanditModel
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.core.contextual_bandit import ContextualBandit
from bandits.data.data_sampler import sample_statlog_data

# ---- 输出目录 ---- #
RESULTS_DIR = os.path.join(ROOT, "results")
DETAILED_DIR = os.path.join(RESULTS_DIR, "detailed_logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
for d in [RESULTS_DIR, DETAILED_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ====================================================================== #
#  模型注册表                                                             #
# ====================================================================== #

def build_registry(custom_model: Optional[str] = None,
                   selected_model: str = "stub",
                   max_new_tokens: int = 256) -> FrozenLLMRegistry:
    reg = FrozenLLMRegistry()
    reg.register("stub", StubFrozenLLM(hidden_dim=128, generate_ds_llm=True))

    preset_models = {
        "smollm2": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "qwen2_5_7b": "/home/csg/Awesome-contextual-bandits/models/huggingface/Qwen2.5-7B-Instruct",
        "llama3_1_8b": "meta-llama/Llama-3.1-8B-Instruct",
    }

    if selected_model in preset_models:
        model_id = preset_models[selected_model]
        # 优先用 MLX (Apple Silicon 2.5x 生成加速)，回退到 PyTorch
        if _HAS_MLX:
            try:
                print(f"[INFO] 加载 {model_id} (MLX) ...")
                reg.register(selected_model, MLXGenerativeFrozenLLM(
                    model_name=model_id,
                    max_new_tokens=max_new_tokens, temperature=0.1,
                    generate_ds_llm=True,
                ), default=True)
            except Exception as e:
                print(f"[WARN] MLX 加载失败: {e}，尝试 PyTorch...")
                if _HAS_TRANSFORMERS:
                    reg.register(selected_model, GenerativeFrozenLLM(
                        model_name=model_id,
                        max_new_tokens=max_new_tokens, temperature=0.1,
                        use_chat_template=True, generate_ds_llm=True,
                    ), default=True)
        elif _HAS_TRANSFORMERS:
            try:
                print(f"[INFO] 加载 {model_id} (PyTorch) ...")
                reg.register(selected_model, GenerativeFrozenLLM(
                    model_name=model_id,
                    max_new_tokens=max_new_tokens, temperature=0.1,
                    use_chat_template=True, generate_ds_llm=True,
                ), default=True)
            except Exception as e:
                print(f"[WARN] 无法加载 {model_id}: {e}")

    if custom_model:
        name = custom_model.split("/")[-1].lower().replace("-", "_")
        if selected_model == name:
            # 自定义模型也优先尝试 MLX
            loaded = False
            if _HAS_MLX:
                try:
                    print(f"[INFO] 加载自定义模型 (MLX): {custom_model} ...")
                    reg.register(name, MLXGenerativeFrozenLLM(
                        model_name=custom_model, max_new_tokens=max_new_tokens,
                        temperature=0.1, generate_ds_llm=True,
                    ), default=True)
                    loaded = True
                except Exception as e:
                    print(f"[WARN] MLX 加载失败: {e}")
            if not loaded and _HAS_TRANSFORMERS:
                try:
                    print(f"[INFO] 加载自定义模型 (PyTorch): {custom_model} ...")
                    reg.register(name, GenerativeFrozenLLM(
                        model_name=custom_model, max_new_tokens=max_new_tokens,
                        temperature=0.1, generate_ds_llm=True,
                    ), default=True)
                except Exception as e:
                    print(f"[ERROR] 无法加载 {custom_model}: {e}")
    return reg


# ====================================================================== #
#  baseline 运行器                                                        #
# ====================================================================== #

def create_baseline(name: str, num_actions: int, context_dim: int) -> Any:
    if name == "neural_bandit":
        hparams = {
            "context_dim": context_dim, "num_actions": num_actions,
            "layer_sizes": [100, 100], "activation": "relu",
            "initial_lr": 0.001, "batch_size": 512, "init_scale": 0.3,
            "use_dropout": False, "dropout_rate": 0.1, "layer_norm": False,
            "verbose": False,
        }
        return NeuralBanditModel(hparams, name="neural_bandit")
    elif name == "neural_linear":
        hparams = {
            "context_dim": context_dim, "num_actions": num_actions,
            "layer_sizes": [100, 100], "activation": "relu",
            "initial_lr": 0.001, "batch_size": 512, "init_scale": 0.3,
            "use_dropout": False, "dropout_rate": 0.1, "layer_norm": False,
            "verbose": False, "lambda_prior": 0.25,
            "a0": 6, "b0": 6, "training_freq": 100,
            "training_freq_network": 100, "training_epochs": 100,
            "initial_pulls": 2,
        }
        return NeuralLinearPosteriorSampling(hparams, name="neural_linear")
    raise ValueError(f"Unknown baseline: {name}")


def run_baseline(
    algo,
    algo_name: str,
    cmab: ContextualBandit,
    offset: int,
    n_rounds: int,
    opt_rewards: np.ndarray,
    opt_actions: np.ndarray,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """逐步运行 baseline 算法。

    Args:
        offset: 数据起始行索引 (= cold_start_n)，保证与 OurMethod 看相同数据。
    """
    actions = np.zeros(n_rounds, dtype=np.int64)
    rewards = np.zeros(n_rounds)

    t0 = time.time()
    for i in range(n_rounds):
        idx = offset + i
        ctx = cmab.context(idx)
        a = algo.action(ctx)
        r = cmab.reward(idx, a)
        algo.update(ctx, a, r)
        actions[i] = a
        rewards[i] = r

        if verbose and i % max(1, n_rounds // 10) == 0:
            cr = np.sum(opt_rewards[:i + 1] - rewards[:i + 1])
            print(f"  [{algo_name}] step {i:>5d}  cum_regret={cr:.1f}")

    elapsed = time.time() - t0
    step_reg = opt_rewards[:n_rounds] - rewards
    cum_reg = np.cumsum(step_reg)
    print(f"  [{algo_name}] done — cum_regret={cum_reg[-1]:.1f}  "
          f"avg={np.mean(rewards):.4f}  time={elapsed:.1f}s")
    return {
        "actions": actions, "rewards": rewards,
        "opt_rewards": opt_rewards[:n_rounds],
        "opt_actions": opt_actions[:n_rounds],
        "step_regret": step_reg,
        "cumulative_regret": cum_reg,
        "cumulative_reward": np.cumsum(rewards),
    }


# ====================================================================== #
#  OurMethod 完整闭环                                                     #
# ====================================================================== #

def run_our_method_pipeline(
    cmab: ContextualBandit,
    opt_rewards: np.ndarray,
    opt_actions: np.ndarray,
    n_rounds: int,
    num_actions: int,
    context_dim: int,
    cold_start_n: int,
    model_name: str = "stub",
    custom_model: Optional[str] = None,
    seed: int = 42,
    z_dim: int = 16,
    c_ucb1: float = 1.0,
    alpha_linucb: float = 1.0,
    c_sim: float = 1.0,
    offline_freq: int = 500,
    offline_epochs: int = 10,
    warmup_epochs: int = 15,
    verbose: bool = True,
    dataset_name: str = "statlog",
    cold_start_tag: Optional[str] = None,
    no_cache: bool = False,
    ckpt: Optional[CheckpointManager] = None,
    arms: Optional[List[Arm]] = None,
    prompt_builder: Optional[StructuredPromptBuilder] = None,
    dump_prompts: int = 0,
    ablation_mode: str = "none",
    run_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """运行 OurMethod 完整闭环流程。

    公平性: 在线阶段使用 cmab.context(cold_start_n + i) 保证与 baseline 看同一组数据。

    参数:
        dataset_name:    数据集名称，用于缓存标签
        cold_start_tag:  指定加载的冷启动缓存标签 (None = 自动检测)
        no_cache:        True = 禁用缓存，总是重新生成
        ckpt:            CheckpointManager 实例 (None = 不保存检查点)
        arms:            自定义 Arm 列表 (None = 使用默认 class_1..class_K)
        prompt_builder:  自定义 PromptBuilder (None = 使用默认通用 builder)
    """
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    import time
    t_start_total = time.time()
    t_cold_start = 0.0
    t_warmup = 0.0
    t_online_infer = 0.0
    t_online_offline_train = 0.0

    # ---- 0.1) 构建注册表 ---- #
    effective_name = model_name
    if custom_model:
        effective_name = custom_model.split("/")[-1].lower().replace("-", "_")

    # ---- 0.2) 尝试从缓存加载冷启动 ---- #
    cache = ColdStartCache()
    use_cache = False
    cache_data = None

    if cold_start_n > 0 and not no_cache:
        tag = cold_start_tag or cache.make_tag(
            dataset_name, effective_name, cold_start_n, seed)
        if cache.exists(tag):
            print(f"[CACHE] 发现冷启动缓存: {tag}")
            cache_data = cache.load(tag)
            use_cache = True
        else:
            print(f"[CACHE] 无缓存 ({tag})，将运行冷启动并保存")

    # ---- 0.3) 构建 LLM (仅在需要时加载) ---- #
    if use_cache:
        # 从缓存获取 h_dim，不需要加载 LLM
        h_dim = cache_data["metadata"]["h_dim"]
        llm = None
        print(f"[OurMethod] 使用缓存 (h_dim={h_dim})，跳过 LLM 加载")
    else:
        reg = build_registry(custom_model=custom_model, selected_model=effective_name)
        if effective_name not in reg:
            print(f"[ERROR] model '{effective_name}' 不可用: {reg.list_models()}")
            return {}
        llm = reg.get(effective_name)
        h_dim = llm.get_hidden_dim()
        print(f"[OurMethod] 模型: {effective_name} (h_dim={h_dim})")

    # ---- 0.4) 构建组件 ---- #
    if arms is None:
        arms = [Arm(arm_id=i, name=f"class_{i + 1}",
                    description=f"Class {i + 1}")
                for i in range(num_actions)]
    pb: StructuredPromptBuilder = prompt_builder or StructuredPromptBuilder(
        role="intelligent decision system",
        scenario=f"{dataset_name} classification as contextual bandit",
        max_feedback=5,
    )
    comp = MLPCompressor(input_dim=h_dim, output_dim=z_dim)
    policy = CombinedUCBPolicy(
        num_actions=num_actions, z_dim=z_dim,
        c_ucb1=c_ucb1, alpha_linucb=alpha_linucb, c_sim=c_sim,
        ablation_mode=ablation_mode,
    )

    # ---- 仿真离线数据集 + 真实离线数据集 ---- #
    sim_dataset = InMemoryOfflineDataset(h_dim=h_dim)
    real_dataset = InMemoryOfflineDataset(h_dim=h_dim)

    # ================================================================== #
    #  Phase 0: 冷启动 (缓存命中则跳过)                                    #
    # ================================================================== #
    if cold_start_n > 0:
        if use_cache:
            # ---- 从缓存恢复 ---- #
            print(f"\n{'=' * 60}")
            print(f"Phase 0: 冷启动 — 从缓存加载 ({cold_start_n} contexts)")
            print(f"{'=' * 60}")

            t0_cache = time.time()
            # 恢复 SimStats
            cache.restore_sim_stats(policy.sim_stats, cache_data)
            # 重建仿真离线数据集
            sim_dataset = cache.build_sim_dataset(cache_data)
            elapsed_cache = time.time() - t0_cache
            print(f"  缓存加载完成: {elapsed_cache:.2f}s")
            print(f"  仿真数据集: {len(sim_dataset)} 条 (从缓存)")
            print(f"  SimStats: {policy.sim_stats.summary()}")
        else:
            # ---- 实际运行冷启动 ---- #
            t0_cold_start = time.time()
            print(f"\n{'=' * 60}")
            print(f"Phase 0: 冷启动 ({cold_start_n} contexts)")
            print(f"{'=' * 60}")

            prev_gen = llm.generate_ds_llm
            llm.generate_ds_llm = True

            # 兼容 cmab.texts，用于传递 env_fields (例如 ag_news 的本文)
            cs_contexts = []
            for i in range(cold_start_n):
                feat = cmab.context(i)
                env_f = {}
                if hasattr(cmab, 'texts') and getattr(cmab, 'texts') is not None:
                    env_f["text"] = str(cmab.texts[i])
                cs_contexts.append(Context(features=feat, env_fields=env_f))
                
            simulator = ColdStartSimulator(llm, pb, arms)
            all_ds, all_h = simulator.warmup_policy(policy, cs_contexts, verbose=verbose)

            # 构建仿真离线数据集
            for i_ctx in range(len(all_ds)):
                ds_list = all_ds[i_ctx]
                h_t = all_h[i_ctx]
                if ds_list:
                    for ds_entry in ds_list:
                        sim_dataset.add(h_t, ds_entry.arm_id, ds_entry.predicted_mean)

            llm.generate_ds_llm = prev_gen
            print(f"  仿真数据集: {len(sim_dataset)} 条 (冷启动)")
            print(f"  SimStats: {policy.sim_stats.summary()}")

            # ---- 保存到缓存供下次使用 ---- #
            tag = cold_start_tag or cache.make_tag(
                dataset_name, effective_name, cold_start_n, seed)
            cache_dir = cache.save(
                tag=tag,
                all_h=all_h,
                all_ds=all_ds,
                sim_stats=policy.sim_stats,
                contexts=[c.features for c in cs_contexts],
                metadata={
                    "dataset": dataset_name,
                    "model": effective_name,
                    "seed": seed,
                    "context_dim": context_dim,
                },
            )
            print(f"  [CACHE] 已保存冷启动缓存: {cache_dir}")
            
            t_cold_start = time.time() - t0_cold_start

    # ================================================================== #
    #  Phase 1: 热启动训练 Compressor                                      #
    # ================================================================== #
    if len(sim_dataset) > 0:
        t0_warmup = time.time()
        print(f"\n{'=' * 60}")
        print(f"Phase 1: 热启动训练 fφ ({warmup_epochs} epochs, {len(sim_dataset)} samples)")
        print(f"{'=' * 60}")

        trainer = OfflineTrainer(comp, policy, lr=1e-3)
        losses = trainer.train(sim_dataset, epochs=warmup_epochs,
                               batch_size=64, verbose=verbose)
        trainer.rebuild_policy_stats(sim_dataset)
        print(f"  热启动 loss: {losses[0]:.6f} → {losses[-1]:.6f}")
        
        t_warmup += time.time() - t0_warmup

        # ---- 检查点: 热启动后 fφ + loss ---- #
        if ckpt:
            ckpt.save_compressor(comp, "warmup")
            ckpt.record_offline_loss("warmup", step=-1,
                                     losses=losses, data_size=len(sim_dataset))

    # ================================================================== #
    #  Phase 2-3: 在线决策 + 周期性离线训练                                 #
    # ================================================================== #
    print(f"\n{'=' * 60}")
    print(f"Phase 2-3: 在线决策 ({n_rounds} rounds, 离线训练每 {offline_freq} 轮)")
    print(f"{'=' * 60}")

    # 在线阶段: 需要 LLM (若之前用缓存跳过了加载，现在按需加载)
    if llm is None:
        reg = build_registry(custom_model=custom_model, selected_model=effective_name)
        if effective_name not in reg:
            print(f"[ERROR] model '{effective_name}' 不可用: {reg.list_models()}")
            return {}
        llm = reg.get(effective_name)
        print(f"[OurMethod] 在线阶段加载 LLM: {effective_name}")

    llm.generate_ds_llm = False

    actions = np.zeros(n_rounds, dtype=np.int64)
    rewards = np.zeros(n_rounds)
    history: List[DecisionRecord] = []

    t0 = time.time()
    for step in range(n_rounds):
        t_step_start = time.time()
        idx = cold_start_n + step
        
        # 临时开启生成以保存 DEBUG
        if step < dump_prompts:
            llm.generate_ds_llm = True

        # --- 1) 获取上下文 --- #
        feat = cmab.context(idx)
        env_f = {}
        if hasattr(cmab, 'texts') and getattr(cmab, 'texts') is not None:
            env_f["text"] = str(cmab.texts[idx])
        ctx = Context(features=feat, env_fields=env_f)

        # --- 2) LLM encode → compress → select --- #
        feedback = history[-5:] if history else None
        prompt = pb.build(ctx, arms, feedback)
        
        h_t, ds_llm = llm.encode(prompt, num_arms=num_actions)
        
        if step < dump_prompts:
            # 恢复状态，避免不小心让整个在线流程全部推理变慢
            llm.generate_ds_llm = False
            
            debug_dir = os.path.join(ROOT, "results", "ourmethod_debug", dataset_name)
            os.makedirs(debug_dir, exist_ok=True)
            dump_path = os.path.join(debug_dir, f"{run_timestamp}_prompts_step{step}.txt")
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write("=== PROMPT ===\n")
                f.write(prompt)
                f.write("\n\n=== DS_LLM OUTPUT ===\n")
                if ds_llm is not None:
                    for d in ds_llm:
                        f.write(f"Arm {d.arm_id}: mean={d.predicted_mean}, std={d.predicted_std}\n")
                else:
                    f.write("None\n")
            if step == dump_prompts - 1:
                print(f"[DEBUG] Dumped {dump_prompts} prompts and ds_llm to {debug_dir}")
        z_t = comp.forward(h_t)
        arm_id = policy.select(ctx, arms, z_t, ds_llm)

        # --- 3) 获取真实奖励 --- #
        r = cmab.reward(idx, arm_id)

        # --- 4) 记录 --- #
        debug = policy.get_last_debug() if hasattr(policy, "get_last_debug") else {}
        
        rec = DecisionRecord(
            step=step, context=ctx, arms=arms, prompt=prompt,
            h_t=h_t, z_t=z_t, ds_llm=ds_llm,
            chosen_arm=arm_id, reward=r,
            optimal_reward=float(opt_rewards[step]),
            regret=float(opt_rewards[step]) - r,
            ucb_values=debug.get("ucb_values"),
            ucb_s1=debug.get("ucb_s1"),
            ucb_s2=debug.get("ucb_s2"),
            ucb_s3=debug.get("ucb_s3"),
            radius_ucb1=debug.get("radius_ucb1"),
            radius_linucb=debug.get("radius_linucb"),
            radius_llm=debug.get("radius_llm"),
            min_source=debug.get("min_source"),
            algorithm=getattr(policy, "name", "combined_ucb"),
        )
        policy.update(rec)
        history.append(rec)
        actions[step] = arm_id
        rewards[step] = r

        # --- 4b) 记录 UCB trace --- #
        if ckpt:
            debug = policy.get_last_debug()
            if debug.get("ucb_s1") is not None:
                ckpt.record_ucb_trace(
                    step=step,
                    chosen_arm=arm_id,
                    reward=r,
                    ucb_s1=debug["ucb_s1"],
                    ucb_s2=debug["ucb_s2"],
                    ucb_s3=debug["ucb_s3"],
                    min_ucb=debug["ucb_values"],
                )

        # --- 5) 积累真实离线数据 --- #
        real_dataset.add(h_t, arm_id, r)

        t_online_infer += time.time() - t_step_start

        # --- 6) 周期性离线训练 --- #
        if (offline_freq > 0
                and (step + 1) % offline_freq == 0
                and len(real_dataset) > 50):
            t_offline_train_start = time.time()
            combined_ds = InMemoryOfflineDataset(h_dim=h_dim)
            
            # =============== [备份原始 回放/累加 逻辑] ===============
            # 之前的方案：在线更新时，会将 Phase 0 的全体仿真数据 (sim_dataset)
            # 与在线收集的真实数据 (real_dataset) 一起混合给压缩器重新训练。
            # for j in range(len(sim_dataset)):
            #     h_j, a_j, r_j = sim_dataset[j]
            #     combined_ds.add(h_j.numpy(), int(a_j), float(r_j))
            # =========================================================

            # [新方针]: 仿真数据仅在冷启动预热阶段使用。
            # 在线运行一旦开启后，只使用现阶段环境真实反馈的记录数据 (real_dataset) 进行训练。
            for j in range(len(real_dataset)):
                h_j, a_j, r_j = real_dataset[j]
                combined_ds.add(h_j.numpy(), int(a_j), float(r_j))
            


            trainer = OfflineTrainer(comp, policy, lr=1e-3)
            losses = trainer.train(combined_ds, epochs=offline_epochs,
                                   batch_size=64, verbose=False)
            trainer.rebuild_policy_stats(combined_ds)

            if verbose:
                cum_r = np.sum(opt_rewards[:step + 1] - rewards[:step + 1])
                print(f"  [step {step + 1:>5d}] offline_train: "
                      f"data={len(combined_ds)}, loss={losses[-1]:.6f}, "
                      f"cum_regret={cum_r:.1f}")

            # ---- 检查点: 周期性离线训练后 fφ + loss ---- #
            if ckpt:
                ckpt.save_compressor(comp, f"step{step + 1}")
                ckpt.record_offline_loss("periodic", step=step + 1,
                                         losses=losses, data_size=len(combined_ds))
                                         
            t_online_offline_train += time.time() - t_offline_train_start

        # --- 7) 进度打印 --- #
        if verbose and step % max(1, n_rounds // 10) == 0:
            cum_r = np.sum(opt_rewards[:step + 1] - rewards[:step + 1])
            print(f"  [combined_ucb] step {step:>5d}  cum_regret={cum_r:.1f}")

    elapsed = time.time() - t0

    # ---- 检查点: 最终状态 ---- #
    if ckpt:
        ckpt.save_compressor(comp, "final")
        ckpt.save_policy_state(policy, "final")
        ckpt.save_offline_losses()
        ckpt.finalize_ucb_trace()

    # ---- 汇总 ---- #
    step_reg = opt_rewards[:n_rounds] - rewards
    cum_reg = np.cumsum(step_reg)
    accuracy = np.mean(actions == opt_actions[:n_rounds])

    print(f"\n  [combined_ucb] done — cum_regret={cum_reg[-1]:.1f}  "
          f"avg={np.mean(rewards):.4f}  acc={accuracy:.4f}  time={elapsed:.1f}s")
    print(f"  SimStats: {policy.sim_stats.summary()}")
    print(f"  真实数据集: {len(real_dataset)} 条; 仿真数据集: {len(sim_dataset)} 条")
    
    t_total = time.time() - t_start_total

    return {
        "actions": actions, "rewards": rewards,
        "opt_rewards": opt_rewards[:n_rounds],
        "opt_actions": opt_actions[:n_rounds],
        "step_regret": step_reg,
        "cumulative_regret": cum_reg,
        "cumulative_reward": np.cumsum(rewards),
        "history": history,
        "times": {
            "total_execution_time": t_total,
            "phase0_cold_start_time": t_cold_start,
            "phase1_warmup_train_time": t_warmup,
            "phase2_online_infer_time": t_online_infer,
            "phase2_periodic_offline_train_time": t_online_offline_train,
        }
    }


# ====================================================================== #
#  保存日志                                                               #
# ====================================================================== #

def save_logs(all_data: Dict[str, Dict[str, np.ndarray]],
              n_rounds: int, seed: int, timestamp: str):
    for algo_name, data in all_data.items():
        fp = os.path.join(DETAILED_DIR,
                          f"statlog_{algo_name}_seed{seed}_{n_rounds}r_{timestamp}.csv")
        df = pd.DataFrame({
            "step": np.arange(n_rounds),
            "action": data["actions"],
            "reward": data["rewards"],
            "opt_action": data["opt_actions"],
            "opt_reward": data["opt_rewards"],
            "is_optimal": (data["actions"] == data["opt_actions"]).astype(int),
            "step_regret": data["step_regret"],
            "cumulative_regret": data["cumulative_regret"],
            "cumulative_reward": data["cumulative_reward"],
        })
        df.to_csv(fp, index=False)
        print(f"  保存: {fp}")

    # 合并 regrets CSV
    regret_fp = os.path.join(DETAILED_DIR,
                             f"statlog_regrets_seed{seed}_{n_rounds}r_{timestamp}.csv")
    rdf = {"step": np.arange(n_rounds)}
    for name, data in all_data.items():
        rdf[f"{name}_step_regret"] = data["step_regret"]
        rdf[f"{name}_cumulative_regret"] = data["cumulative_regret"]
    pd.DataFrame(rdf).to_csv(regret_fp, index=False)
    print(f"  合并遗憾: {regret_fp}")

    # 合并 rewards CSV
    reward_fp = os.path.join(DETAILED_DIR,
                             f"statlog_rewards_seed{seed}_{n_rounds}r_{timestamp}.csv")
    rwdf = {"step": np.arange(n_rounds)}
    for name, data in all_data.items():
        rwdf[f"{name}_reward"] = data["rewards"]
        rwdf[f"{name}_cumulative_reward"] = data["cumulative_reward"]
    first = next(iter(all_data.values()))
    rwdf["optimal_reward"] = first["opt_rewards"]
    pd.DataFrame(rwdf).to_csv(reward_fp, index=False)
    print(f"  合并奖励: {reward_fp}")


# ====================================================================== #
#  绘图                                                                   #
# ====================================================================== #

def plot_curves(all_data: Dict[str, Dict[str, np.ndarray]],
                n_rounds: int, seed: int, timestamp: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    steps = np.arange(n_rounds)
    styles = {
        "neural_bandit":  {"color": "#1f77b4", "ls": "--",  "lw": 2},
        "neural_linear":  {"color": "#ff7f0e", "ls": "-.",  "lw": 2},
        "combined_ucb":   {"color": "#2ca02c", "ls": "-",   "lw": 2.5},
    }

    # ---- 累积遗憾 ---- #
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in all_data.items():
        s = styles.get(name, {"color": "gray", "ls": "-", "lw": 1})
        ax.plot(steps, data["cumulative_regret"], label=name, **s)
    ax.set_xlabel("Time Step", fontsize=13)
    ax.set_ylabel("Cumulative Regret", fontsize=13)
    ax.set_title(f"Statlog — Cumulative Regret ({n_rounds} rounds, seed={seed})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(PLOTS_DIR,
                      f"statlog_cumulative_regret_seed{seed}_{n_rounds}r_{timestamp}.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存遗憾曲线: {fp}")

    # ---- 累积奖励 ---- #
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in all_data.items():
        s = styles.get(name, {"color": "gray", "ls": "-", "lw": 1})
        ax.plot(steps, data["cumulative_reward"], label=name, **s)
    opt_cum = np.cumsum(next(iter(all_data.values()))["opt_rewards"])
    ax.plot(steps, opt_cum, label="optimal", color="red", ls=":", lw=2)
    ax.set_xlabel("Time Step", fontsize=13)
    ax.set_ylabel("Cumulative Reward", fontsize=13)
    ax.set_title(f"Statlog — Cumulative Reward ({n_rounds} rounds, seed={seed})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(PLOTS_DIR,
                      f"statlog_cumulative_reward_seed{seed}_{n_rounds}r_{timestamp}.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存奖励曲线: {fp}")

    # ---- 滑动窗口 ---- #
    fig, ax = plt.subplots(figsize=(12, 6))
    window = min(200, n_rounds // 5)
    for name, data in all_data.items():
        s = styles.get(name, {"color": "gray", "ls": "-", "lw": 1})
        rolling = pd.Series(data["rewards"]).rolling(window, min_periods=1).mean()
        ax.plot(steps, rolling, label=name, **s)
    ax.axhline(y=1.0, color="red", ls=":", lw=1.5, label="optimal (=1.0)")
    ax.set_xlabel("Time Step", fontsize=13)
    ax.set_ylabel(f"Average Reward (window={window})", fontsize=13)
    ax.set_title(f"Statlog — Rolling Average Reward ({n_rounds} rounds, seed={seed})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fp = os.path.join(PLOTS_DIR,
                      f"statlog_rolling_reward_seed{seed}_{n_rounds}r_{timestamp}.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存滑动奖励曲线: {fp}")


# ====================================================================== #
#  汇总                                                                   #
# ====================================================================== #

def print_summary(all_data: Dict[str, Dict[str, np.ndarray]], n_rounds: int):
    print(f"\n{'=' * 70}")
    print(f"{'算法':<20s} {'累计遗憾':>12s} {'平均奖励':>12s} {'准确率':>10s}")
    print(f"{'=' * 70}")
    for name, data in all_data.items():
        cr = data["cumulative_regret"][-1]
        avg_r = np.mean(data["rewards"])
        acc = np.mean(data["actions"] == data["opt_actions"])
        print(f"{name:<20s} {cr:>12.1f} {avg_r:>12.4f} {acc:>10.4f}")
    print(f"{'=' * 70}")

    seg = max(n_rounds // 5, 1)
    print(f"\n分段平均奖励 (每段 {seg} 轮):")
    header = f"{'区间':<16s}"
    for name in all_data:
        header += f" {name:>16s}"
    print(header)
    print("-" * len(header))
    for s_idx in range(5):
        lo, hi = s_idx * seg, min((s_idx + 1) * seg, n_rounds)
        row = f"[{lo:>5d}-{hi:>5d}]    "
        for name, data in all_data.items():
            avg = np.mean(data["rewards"][lo:hi])
            row += f" {avg:>16.4f}"
        print(row)


# ====================================================================== #
#  主入口                                                                 #
# ====================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="OurMethod Full Pipeline: cold-start → warm-up → online → offline → compare")
    parser.add_argument("--model", dest="model_name", type=str, default="stub",
                        help="LLM model: stub | smollm2 | qwen2_5_7b | llama3_1_8b")
    parser.add_argument("--custom_model", type=str, default=None,
                        help="HuggingFace model ID (e.g. HuggingFaceTB/SmolLM2-360M-Instruct)")
    parser.add_argument("--n_rounds", type=int, default=5000,
                        help="在线决策轮数")
    parser.add_argument("--cold_start_n", type=int, default=50,
                        help="冷启动仿真上下文数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--c_ucb1", type=float, default=1.0)
    parser.add_argument("--alpha_linucb", type=float, default=1.0)
    parser.add_argument("--c_sim", type=float, default=1.0)
    parser.add_argument("--offline_freq", type=int, default=500,
                        help="每 N 轮离线训练 (0=不离线)")
    parser.add_argument("--offline_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--dataset", type=str, default="statlog",
                        help="数据集名称 (用于缓存标签)")
    parser.add_argument("--cold_start_tag", type=str, default=None,
                        help="加载已有冷启动缓存的标签 (为空则自动检测/生成)")
    parser.add_argument("--no_cache", action="store_true",
                        help="禁用冷启动缓存 (总是重新生成)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    n_rounds = args.n_rounds
    cold_start_n = args.cold_start_n
    seed = args.seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    np.random.seed(seed)

    # ---- 创建 CheckpointManager ---- #
    exp_tag = (f"{args.dataset}_{args.model_name}_{n_rounds}r"
               f"_seed{seed}_{timestamp}")
    ckpt = CheckpointManager(experiment_tag=exp_tag)
    ckpt.save_config({
        "dataset": args.dataset,
        "model": args.model_name,
        "custom_model": args.custom_model,
        "n_rounds": n_rounds,
        "cold_start_n": cold_start_n,
        "seed": seed,
        "z_dim": args.z_dim,
        "c_ucb1": args.c_ucb1,
        "alpha_linucb": args.alpha_linucb,
        "c_sim": args.c_sim,
        "offline_freq": args.offline_freq,
        "offline_epochs": args.offline_epochs,
        "warmup_epochs": args.warmup_epochs,
        "cold_start_tag": args.cold_start_tag,
        "no_cache": args.no_cache,
    })
    print(f"[CKPT] 检查点目录: {ckpt.dir}")

    # ---- 1) 采样 Statlog ---- #
    data_path = os.path.join(ROOT, "datasets", "statlog.trn")
    if not os.path.exists(data_path):
        print(f"[ERROR] 找不到 {data_path}")
        sys.exit(1)

    total_needed = min(n_rounds + cold_start_n, 43500)
    dataset, (opt_rewards_all, opt_actions_all) = sample_statlog_data(
        data_path, total_needed, shuffle_rows=True,
    )
    num_actions = 7
    context_dim = dataset.shape[1] - num_actions

    # opt_rewards / opt_actions 对齐到 [cold_start_n, cold_start_n+n_rounds)
    opt_rewards = opt_rewards_all[cold_start_n:cold_start_n + n_rounds]
    opt_actions = opt_actions_all[cold_start_n:cold_start_n + n_rounds]

    print(f"Statlog: {total_needed} 行, {context_dim} 维特征, {num_actions} 动作")
    print(f"cold_start_n={cold_start_n}, n_rounds={n_rounds}")
    print(f"seed={seed}, offline_freq={args.offline_freq}, offline_epochs={args.offline_epochs}")
    print(f"model={args.model_name}, timestamp={timestamp}")
    print()

    cmab = ContextualBandit(num_actions, context_dim)
    cmab.feed_data(dataset)

    all_data: Dict[str, Dict[str, np.ndarray]] = {}

    # ---- 2) neural_bandit ---- #
    print("=" * 60)
    print("运行 neural_bandit ...")
    np.random.seed(seed)
    algo_nb = create_baseline("neural_bandit", num_actions, context_dim)
    all_data["neural_bandit"] = run_baseline(
        algo_nb, "neural_bandit", cmab,
        offset=cold_start_n, n_rounds=n_rounds,
        opt_rewards=opt_rewards, opt_actions=opt_actions,
        verbose=args.verbose,
    )

    # ---- 3) neural_linear ---- #
    print("\n" + "=" * 60)
    print("运行 neural_linear ...")
    np.random.seed(seed)
    algo_nl = create_baseline("neural_linear", num_actions, context_dim)
    all_data["neural_linear"] = run_baseline(
        algo_nl, "neural_linear", cmab,
        offset=cold_start_n, n_rounds=n_rounds,
        opt_rewards=opt_rewards, opt_actions=opt_actions,
        verbose=args.verbose,
    )

    # ---- 4) OurMethod ---- #
    print("\n" + "=" * 60)
    print("运行 combined_ucb (OurMethod Full Pipeline) ...")
    np.random.seed(seed)
    our_result = run_our_method_pipeline(
        cmab=cmab,
        opt_rewards=opt_rewards,
        opt_actions=opt_actions,
        n_rounds=n_rounds,
        num_actions=num_actions,
        context_dim=context_dim,
        cold_start_n=cold_start_n,
        model_name=args.model_name,
        custom_model=args.custom_model,
        seed=seed,
        z_dim=args.z_dim,
        c_ucb1=args.c_ucb1,
        alpha_linucb=args.alpha_linucb,
        c_sim=args.c_sim,
        offline_freq=args.offline_freq,
        offline_epochs=args.offline_epochs,
        warmup_epochs=args.warmup_epochs,
        verbose=args.verbose,
        dataset_name=args.dataset,
        cold_start_tag=args.cold_start_tag,
        no_cache=args.no_cache,
        ckpt=ckpt,
    )
    if our_result:
        all_data["combined_ucb"] = our_result

    # ---- 5) 保存日志 ---- #
    print("\n" + "=" * 60)
    print("保存日志 ...")
    save_logs(all_data, n_rounds, seed, timestamp)

    # ---- 6) 汇总 ---- #
    print_summary(all_data, n_rounds)

    # ---- 7) 绘图 ---- #
    print("\n绘制曲线 ...")
    plot_curves(all_data, n_rounds, seed, timestamp)

    # ---- 8) 保存汇总 JSON ---- #
    import json
    summary_path = os.path.join(RESULTS_DIR,
                                f"results_{n_rounds}r_{timestamp}.json")
    summary = {}
    for name, data in all_data.items():
        summary[name] = {
            "cumulative_regret": float(data["cumulative_regret"][-1]),
            "average_reward": float(np.mean(data["rewards"])),
            "accuracy": float(np.mean(data["actions"] == data["opt_actions"])),
        }
    summary["meta"] = {
        "n_rounds": n_rounds, "cold_start_n": cold_start_n,
        "seed": seed, "model": args.model_name,
        "offline_freq": args.offline_freq,
        "offline_epochs": args.offline_epochs,
        "warmup_epochs": args.warmup_epochs,
        "timestamp": timestamp,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总 JSON: {summary_path}")

    # ---- 9) CheckpointManager 汇总 ---- #
    ckpt.save_experiment_summary(extra={
        "results": summary,
        "timestamp": timestamp,
    })
    print(f"检查点汇总: {ckpt.dir}/experiment_summary.json")

    print(f"\n完成! 结果保存在 {RESULTS_DIR}/")
    print(f"检查点保存在 {ckpt.dir}/")


if __name__ == "__main__":
    main()

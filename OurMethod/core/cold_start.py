"""冷启动仿真器：在在线交互前，用冻结 LLM 为一组上下文生成奖励预测。

预填充 SimulationStats 的 sum_llm / t_s，使 UCB^s3 在第 0 步就有约束力。
同时收集 (h_t, ds_llm) 用于 Compressor 的 warm-up 热启动训练。

用法::

    simulator = ColdStartSimulator(llm, prompt_builder, arms)
    all_ds, all_h = simulator.warmup_policy(policy, contexts, verbose=True)
    # → policy.sim_stats 已预填充，UCB^s3 第 0 步即可生效
    # → all_h 可用于 OfflineTrainer 热启动训练
"""

from __future__ import annotations

import sys
import os
import time

_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

from typing import List, Optional

import numpy as np

from OurMethod.core.protocol import Arm, Context, DecisionRecord, DsLlm, PromptBuilder, FrozenLLMEncoder
from OurMethod.core.sim_stats import SimulationStats


class ColdStartSimulator:
    """冷启动仿真器。

    在在线交互开始前，对一组上下文调用冻结 LLM 获取 ds_llm 预测，
    将预测写入 Policy 的 sim_stats，使 UCB^s3 从第 0 轮就有意义。

    Args:
        llm_encoder:    冻结 LLM 编码器（必须 generate_ds_llm=True）
        prompt_builder: Prompt 生成器
        arms:           候选动作列表

    # ============================================================ #
    #  参数指南 (Parameter Guide)
    # ============================================================ #
    #
    # 🔴 重点关注
    #
    #   cold_start_n [在 warmup_policy/simulate 的 contexts 长度中体现]
    #       冷启动仿真的上下文数量（由外部调用方传入）。
    #       增大 → UCB^s3 更准确，但 LLM 生成耗时线性增长。
    #       每个上下文约耗时 ~8s (SmolLM2, 含生成)。
    #       典型范围: [20, 200]。50 已足以提供有效先验。
    #       若 LLM 质量高 → 增大; 若冷启动时间受限 → 减小即可。
    #
    # 🟢 无需调整
    #
    #   llm_encoder, prompt_builder, arms — 由外部组装传入
    # ============================================================ #
    """

    def __init__(
        self,
        llm_encoder: FrozenLLMEncoder,
        prompt_builder: PromptBuilder,
        arms: List[Arm],
    ):
        self.llm = llm_encoder
        self.pb = prompt_builder
        self.arms = arms

    def simulate(
        self,
        contexts: List[np.ndarray],
        verbose: bool = False,
    ) -> tuple:
        """对每个上下文调用冻结 LLM，返回 (ds_llm 列表, h_t 列表)。

        Args:
            contexts: 特征向量列表，每个 shape (d,)
            verbose:  是否打印进度

        Returns:
            (all_ds, all_h):
              all_ds: 长度 = len(contexts) 的列表，每项为 List[DsLlm] 或 None
              all_h:  长度 = len(contexts) 的列表，每项为 np.ndarray (h_dim,)
        """
        all_ds: List[Optional[List[DsLlm]]] = []
        all_h: List[np.ndarray] = []
        t0 = time.time()
        n = len(contexts)

        for i, feat in enumerate(contexts):
            ctx = Context(features=feat)
            prompt = self.pb.build(ctx, self.arms)
            h_t, ds_llm = self.llm.encode(prompt, num_arms=len(self.arms))
            all_ds.append(ds_llm)
            all_h.append(h_t)

            if verbose and (i + 1) % max(1, n // 10) == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n - i - 1)
                print(f"  Cold-start: {i+1}/{n}  "
                      f"elapsed={elapsed:.1f}s  eta={eta:.1f}s")

        if verbose:
            print(f"  Cold-start complete: {n} contexts in {time.time()-t0:.1f}s")

        return all_ds, all_h

    def warmup_policy(
        self,
        policy,
        contexts: List[np.ndarray],
        verbose: bool = False,
    ) -> tuple:
        """用 LLM 预测预填充策略的 sim_stats。

        Args:
            policy:   Must have .sim_stats: SimulationStats 属性
            contexts: 特征向量列表
            verbose:  是否打印进度

        Returns:
            (all_ds, all_h):
              all_ds: simulate() 的 ds_llm 返回值
              all_h:  simulate() 的 h_t 返回值（用于 warm-up 训练）
        """
        all_ds, all_h = self.simulate(contexts, verbose=verbose)

        for ds in all_ds:
            if ds:
                if hasattr(policy, "sim_stats"):
                    policy.sim_stats.update_from_ds_llm(ds)

        if verbose:
            stats = policy.sim_stats.summary() if hasattr(policy, "sim_stats") else {}
            print(f"  Warmup 完成: T_s = {stats.get('t_s', 'N/A')}")

        return all_ds, all_h

    def warmup_sim_stats(
        self,
        sim_stats: SimulationStats,
        contexts: List[np.ndarray],
        verbose: bool = False,
    ) -> tuple:
        """直接预填充 SimulationStats 对象（不需要完整 policy）。"""
        all_ds, all_h = self.simulate(contexts, verbose=verbose)

        for ds in all_ds:
            if ds:
                sim_stats.update_from_ds_llm(ds)

        return all_ds, all_h

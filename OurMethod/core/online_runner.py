"""Step 1 — 在线运行器：把 PromptBuilder / FrozenLLM / Compressor / Policy 串成闭环。

OnlineRunner 不绑定任何具体实现，只依赖四大接口。
每轮输出一条 DecisionRecord → 追加写入 decisions.jsonl。
"""

from __future__ import annotations

import json
import os
import sys
import time
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from OurMethod.core.protocol import (
    Arm, Context, DecisionRecord, DsLlm,
    PromptBuilder, FrozenLLMEncoder, Compressor, Policy,
)


class OnlineRunner:
    """在线上下文赌博机实验运行器。

    依赖：
      env_reward_fn(context_features, arm_id) -> float
      env_optimal_fn(context_features) -> (opt_reward, opt_arm)  （可选）
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        llm_encoder: FrozenLLMEncoder,
        compressor: Compressor,
        policy: Policy,
        arms: List[Arm],
        env_reward_fn: Callable[[np.ndarray, int], float],
        env_optimal_fn: Optional[Callable[[np.ndarray], Tuple[float, int]]] = None,
        context_sampler: Optional[Callable[[], np.ndarray]] = None,
        context_dim: int = 10,
        log_path: Optional[str] = None,
        max_feedback: int = 10,
        seed: int = 42,
    ):
        # ============================================================ #
        #  参数指南 (Parameter Guide)
        # ============================================================ #
        #
        # 🔴 重点关注
        #
        #   n_rounds [在 run() 中] (int)
        #       在线实验总轮次。决定数据量和 UCB 收敛程度。
        #       典型范围: [500, 50000]。轮次越多 LinUCB 越准。
        #
        # 🟡 可按需调整
        #
        #   max_feedback (int, default=10)
        #       每轮 prompt 中附带的历史反馈条数。
        #       与 PromptBuilder.max_feedback 配合使用。
        #
        #   seed (int, default=42)
        #       随机种子，控制上下文采样的可复现性。
        #
        #   context_dim (int, default=10)
        #       默认上下文采样器的特征维度（自定义 sampler 时忽略）。
        #
        # 🟢 无需调整
        #
        #   其余参数为组件注入（prompt_builder, llm_encoder,
        #   compressor, policy, arms, env_reward_fn 等），
        #   由外部实验脚本统一组装。
        # ============================================================ #
        self.pb = prompt_builder
        self.llm = llm_encoder
        self.comp = compressor
        self.policy = policy
        self.arms = arms
        self.reward_fn = env_reward_fn
        self.optimal_fn = env_optimal_fn
        self.max_feedback = max_feedback
        self.rng = np.random.RandomState(seed)
        self.context_dim = context_dim

        if context_sampler is not None:
            self._sample_ctx = context_sampler
        else:
            self._sample_ctx = lambda: self.rng.randn(self.context_dim)

        self.log_path = log_path
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

        self.history: List[DecisionRecord] = []

    # ================================================================== #
    def run(self, n_rounds: int, verbose: bool = False) -> List[DecisionRecord]:
        """运行 n_rounds 轮在线决策。"""
        log_fh = None
        if self.log_path:
            log_fh = open(self.log_path, "a")

        for step in range(n_rounds):
            # 1) 采样上下文
            feat = self._sample_ctx()
            ctx = Context(features=feat)

            # 2) 生成 prompt
            feedback = self.history[-self.max_feedback:] if self.history else None
            prompt = self.pb.build(ctx, self.arms, feedback)

            # 3) 冻结 LLM 编码
            h_t, ds_llm = self.llm.encode(prompt, num_arms=len(self.arms))

            # 4) 特征压缩
            z_t = self.comp.forward(h_t)

            # 5) 选择 arm
            arm_id = self.policy.select(ctx, self.arms, z_t, ds_llm)

            # 6) 获取 reward
            reward = self.reward_fn(feat, arm_id)

            # 7) 最优奖励（可选）
            opt_reward, opt_arm = (None, None)
            if self.optimal_fn:
                opt_reward, opt_arm = self.optimal_fn(feat)

            # 8) 构建 DecisionRecord
            debug = {}
            if hasattr(self.policy, "get_last_debug"):
                debug = self.policy.get_last_debug()

            rec = DecisionRecord(
                step=len(self.history),
                context=ctx,
                arms=self.arms,
                prompt=prompt,
                h_t=h_t,
                z_t=z_t,
                ds_llm=ds_llm,
                chosen_arm=arm_id,
                reward=reward,
                optimal_reward=opt_reward,
                regret=(opt_reward - reward) if opt_reward is not None else None,
                ucb_values=debug.get("ucb_values"),
                radius_ucb1=debug.get("radius_ucb1"),
                radius_linucb=debug.get("radius_linucb"),
                radius_llm=debug.get("radius_llm"),
                algorithm=getattr(self.policy, "name", "combined_ucb"),
                timestamp=time.time(),
            )

            # 9) 更新策略
            self.policy.update(rec)
            self.history.append(rec)

            # 10) 写日志
            if log_fh:
                log_fh.write(rec.to_json() + "\n")
                log_fh.flush()

            if verbose and step % max(1, n_rounds // 10) == 0:
                cum_regret = sum(r.regret for r in self.history if r.regret is not None)
                print(f"  [step {step:>5d}] cum_regret={cum_regret:.2f}")

        if log_fh:
            log_fh.close()

        return self.history

    # ================================================================== #
    def summary(self) -> Dict[str, float]:
        rewards = [r.reward for r in self.history]
        regrets = [r.regret for r in self.history if r.regret is not None]
        return {
            "rounds": len(self.history),
            "cumulative_reward": sum(rewards),
            "cumulative_regret": sum(regrets) if regrets else 0.0,
            "average_reward": float(np.mean(rewards)) if rewards else 0.0,
        }

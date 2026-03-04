"""Step 2 — 结构化 Prompt 生成器（通用版）。

Prompt 分 5 段（用关键字段标记，便于自检断言 & LLM 解析）：
  [ROLE]      角色定义 + 决策目标
  [CONTEXT]   用户/设备/特征向量
  [ARMS]      候选动作列表及描述
  [FEEDBACK]  最近 N 条交互反馈（可选）
  [TASK]      任务指令 + JSON 输出格式

新增:
  build_messages()  返回 chat 消息列表 [{role, content}, ...]
                    可直接传入 tokenizer.apply_chat_template()。
"""

from __future__ import annotations

import sys, os
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

from typing import Dict, List, Optional

import numpy as np

from OurMethod.core.protocol import (
    Arm, Context, DecisionRecord, DsLlm, PromptBuilder,
)


class StructuredPromptBuilder(PromptBuilder):
    """按文档要求生成可复现、可解析的结构化 Prompt（通用版，不绑定特定数据集）。"""

    def __init__(
        self,
        role: str = "intelligent decision system",
        scenario: str = "contextual bandit optimization",
        max_feedback: int = 10,
        feature_names: Optional[List[str]] = None,
        arm_reward_range: str = "0-1",
        max_feature_display: int = 15,
    ):
        # ============================================================ #
        #  参数指南 (Parameter Guide)
        # ============================================================ #
        #
        # 🟡 可按需调整
        #
        #   role (str, default="intelligent decision system")
        #       角色定义，写入 [ROLE] 段。
        #       可根据具体任务自定义，如 "shuttle classification expert"。
        #
        #   scenario (str, default="contextual bandit optimization")
        #       场景描述，写入 [ROLE] 段。
        #       可根据数据集自定义，帮助 LLM 理解任务背景。
        #
        #   max_feedback (int, default=10)
        #       [FEEDBACK] 段最多包含的历史交互条数。
        #       增大 → prompt 更长，LLM 获得更多上下文但推理更慢。
        #       典型范围: [3, 20]。
        #
        #   feature_names (list, optional)
        #       特征名称列表。若提供，[CONTEXT] 段显示 "名称=值" 而非纯向量。
        #       有名称时 LLM 能更好理解特征含义，建议尽量提供。
        #
        #   arm_reward_range (str, default="0-1")
        #       奖励范围描述。帮助 LLM 在 ds_llm 中给出合理范围的预测。
        #
        #   max_feature_display (int, default=15)
        #       无特征名称时，prompt 中最多显示的特征数量。
        #       减小可缩短 prompt，加快 LLM 编码/生成速度。
        #       典型范围: [5, 30]。高维特征建议设 10-15。
        # ============================================================ #
        self.role = role
        self.scenario = scenario
        self.max_feedback = max_feedback
        self.feature_names = feature_names
        self.arm_reward_range = arm_reward_range
        self.max_feature_display = max_feature_display

    # ------------------------------------------------------------------ #
    def build(
        self,
        context: Context,
        arms: List[Arm],
        last_feedback: Optional[List[DecisionRecord]] = None,
    ) -> str:
        sections = [
            self._sec_role(),
            self._sec_context(context),
            self._sec_arms(arms),
        ]
        if last_feedback:
            sections.append(self._sec_feedback(last_feedback))
        sections.append(self._sec_task(arms))
        return "\n\n".join(sections)

    def build_messages(
        self,
        context: Context,
        arms: List[Arm],
        last_feedback: Optional[List[DecisionRecord]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """构建 chat 消息列表（system + user），可传入 chat_template。"""
        user_content = self.build(context, arms, last_feedback)
        msgs: List[Dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_content})
        return msgs

    # ------------------------------------------------------------------ #
    #  段落生成
    # ------------------------------------------------------------------ #
    def _sec_role(self) -> str:
        return (
            "[ROLE]\n"
            f"You are a {self.role} performing {self.scenario}.\n"
            "Your goal is to maximize cumulative reward over sequential decisions.\n"
            "Analyze the context carefully and predict which action yields the highest reward."
        )

    def _sec_context(self, ctx: Context) -> str:
        lines = ["[CONTEXT]"]
        if ctx.user_id:
            lines.append(f"User ID: {ctx.user_id}")
        if ctx.machine_id:
            lines.append(f"Machine ID: {ctx.machine_id}")
        if ctx.env_fields:
            for k, v in sorted(ctx.env_fields.items()):
                lines.append(f"{k}: {v}")
        feats = ctx.features
        if self.feature_names and len(self.feature_names) == len(feats):
            for n, v in zip(self.feature_names, feats):
                lines.append(f"  {n} = {v:.6f}")
        else:
            nd = self.max_feature_display
            lines.append(f"Feature vector (dim={len(feats)}): "
                         + ", ".join(f"{v:.4f}" for v in feats[:nd]))
            if len(feats) > nd:
                lines.append(f"  ... ({len(feats)-nd} more dimensions)")
        if ctx.stat_features is not None:
            lines.append(f"Stat features: "
                         + ", ".join(f"{v:.4f}" for v in ctx.stat_features[:10]))
        return "\n".join(lines)

    def _sec_arms(self, arms: List[Arm]) -> str:
        lines = ["[ARMS]"]
        for a in sorted(arms, key=lambda x: x.arm_id):
            name = a.name or f"arm_{a.arm_id}"
            desc = f" — {a.description}" if a.description else ""
            lines.append(f"  Action {a.arm_id} ({name}){desc}")
            if a.attributes:
                for k, v in sorted(a.attributes.items()):
                    lines.append(f"    {k}: {v}")
        return "\n".join(lines)

    def _sec_feedback(self, records: List[DecisionRecord]) -> str:
        recent = records[-self.max_feedback:]
        lines = [f"[FEEDBACK] (last {len(recent)} rounds)"]
        for r in recent:
            line = (f"  Step {r.step}: chose arm {r.chosen_arm}, "
                    f"reward={r.reward:.4f}")
            if r.regret is not None:
                line += f", regret={r.regret:.4f}"
            lines.append(line)
        return "\n".join(lines)

    def _sec_task(self, arms: List[Arm]) -> str:
        arm_ids = ", ".join(str(a.arm_id) for a in arms)
        n = len(arms)
        return (
            "[TASK]\n"
            f"For ALL {n} actions below, predict expected reward and uncertainty.\n"
            f"You MUST include exactly {n} objects, one per arm_id: {arm_ids}.\n"
            "\n"
            "Output ONLY a JSON array (no extra text):\n"
            "[\n"
            '  {"arm_id": <int>, "predicted_mean": <float>, '
            '"predicted_std": <float>, "reasoning": "<str>"},\n'
            "  ...\n"
            "]\n"
            f"\nCandidate arm IDs: [{arm_ids}]"
        )

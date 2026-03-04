"""Step 2 自检 — 结构化 Prompt 验证。

Done 标准：
  ✓ Prompt 包含 [ROLE] 段
  ✓ Prompt 包含 [CONTEXT] 段（带 User ID / Feature vector）
  ✓ Prompt 包含 [ARMS] 段（每个 arm 出现 arm_id）
  ✓ Prompt 包含 [FEEDBACK] 段（当提供历史时）
  ✓ Prompt 包含 [TASK] 段（含 JSON 输出格式要求）
  ✓ 相同输入 → 相同 Prompt（确定性）
"""

import sys, os, unittest
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np

from OurMethod.core.protocol import Arm, Context, DecisionRecord
from OurMethod.core.prompt_builder import StructuredPromptBuilder


def _make_context() -> Context:
    return Context(
        features=np.array([0.1, 0.2, 0.3]),
        user_id="user_42",
        machine_id="machine_7",
        env_fields={"temperature": 22.5},
    )


def _make_arms() -> list:
    return [
        Arm(arm_id=0, name="alpha", description="First option"),
        Arm(arm_id=1, name="beta", description="Second option"),
        Arm(arm_id=2, name="gamma"),
    ]


def _make_feedback() -> list:
    ctx = _make_context()
    arms = _make_arms()
    return [
        DecisionRecord(step=0, context=ctx, arms=arms, prompt="p",
                        chosen_arm=1, reward=0.7),
        DecisionRecord(step=1, context=ctx, arms=arms, prompt="p",
                        chosen_arm=0, reward=0.3, optimal_reward=0.8),
    ]


class TestPromptSections(unittest.TestCase):
    def setUp(self):
        self.pb = StructuredPromptBuilder()
        self.ctx = _make_context()
        self.arms = _make_arms()
        self.prompt = self.pb.build(self.ctx, self.arms)

    def test_role_section(self):
        self.assertIn("[ROLE]", self.prompt)

    def test_context_section(self):
        self.assertIn("[CONTEXT]", self.prompt)
        self.assertIn("user_42", self.prompt)

    def test_arms_section(self):
        self.assertIn("[ARMS]", self.prompt)
        for a in self.arms:
            self.assertIn(f"Action {a.arm_id}", self.prompt)

    def test_task_section(self):
        self.assertIn("[TASK]", self.prompt)
        self.assertIn("predicted_mean", self.prompt)
        self.assertIn("predicted_std", self.prompt)
        self.assertIn("JSON", self.prompt)

    def test_no_feedback_section_when_empty(self):
        # 没有反馈时不出现 FEEDBACK 段
        self.assertNotIn("[FEEDBACK]", self.prompt)

    def test_feedback_section_when_provided(self):
        fb = _make_feedback()
        prompt_fb = self.pb.build(self.ctx, self.arms, fb)
        self.assertIn("[FEEDBACK]", prompt_fb)
        self.assertIn("reward=", prompt_fb)


class TestPromptDeterminism(unittest.TestCase):
    def test_same_input_same_output(self):
        pb = StructuredPromptBuilder()
        ctx = _make_context()
        arms = _make_arms()
        p1 = pb.build(ctx, arms)
        p2 = pb.build(ctx, arms)
        self.assertEqual(p1, p2, "相同输入应产生相同 Prompt")


if __name__ == "__main__":
    unittest.main()

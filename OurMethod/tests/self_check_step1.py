"""Step 1 自检 — 在线运行器闭环验证。

Done 标准：
  ✓ 100 轮在线运行无异常
  ✓ UCB1 计数总和 == 100
  ✓ LinUCB A 矩阵不再是初始值（至少有一个 arm 被更新过）
  ✓ decisions.jsonl 行数 == 100
  ✓ 每条记录含 prompt / h_t / z_t / chosen_arm / reward 字段
"""

import sys, os, unittest, json, tempfile
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np

from OurMethod.core.protocol import Arm, Context, DecisionRecord
from OurMethod.core.prompt_builder import StructuredPromptBuilder
from OurMethod.core.frozen_llm import StubFrozenLLM
from OurMethod.core.compressor import MLPCompressor
from OurMethod.core.combined_policy import CombinedUCBPolicy
from OurMethod.core.online_runner import OnlineRunner
from OurMethod.core.env_simulated import SimulatedEnvironment


N_ROUNDS = 100
K, D = 5, 10


def _build_runner(log_path: str) -> OnlineRunner:
    env = SimulatedEnvironment(num_actions=K, context_dim=D, seed=42)
    return OnlineRunner(
        prompt_builder=StructuredPromptBuilder(),
        llm_encoder=StubFrozenLLM(hidden_dim=32, generate_ds_llm=True),
        compressor=MLPCompressor(input_dim=32, output_dim=8),
        policy=CombinedUCBPolicy(num_actions=K, z_dim=8),
        arms=[Arm(arm_id=i, name=f"arm_{i}") for i in range(K)],
        env_reward_fn=env.reward,
        env_optimal_fn=env.optimal_reward,
        context_dim=D,
        log_path=log_path,
        seed=42,
    )


class TestOnlineRunner(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._log = os.path.join(self._tmpdir, "decisions.jsonl")
        self.runner = _build_runner(self._log)
        self.history = self.runner.run(N_ROUNDS)

    def test_history_length(self):
        self.assertEqual(len(self.history), N_ROUNDS)

    def test_ucb1_total_count(self):
        policy = self.runner.policy
        self.assertAlmostEqual(policy.counts.sum(), N_ROUNDS, places=0)

    def test_linucb_A_changed(self):
        """至少有一个 arm 的 A 矩阵不再是初始 λI。"""
        policy = self.runner.policy
        changed = False
        for a in range(K):
            init_A = policy.lambda_reg * np.eye(policy.z_dim)
            if not np.allclose(policy.A[a], init_A):
                changed = True
                break
        self.assertTrue(changed, "所有 arm 的 A 矩阵都没变化")

    def test_jsonl_line_count(self):
        with open(self._log) as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), N_ROUNDS)

    def test_jsonl_required_fields(self):
        required = {"step", "prompt", "h_t", "z_t", "chosen_arm", "reward"}
        with open(self._log) as f:
            for idx, line in enumerate(f):
                d = json.loads(line)
                for k in required:
                    self.assertIn(k, d, f"第 {idx} 行缺少字段 {k}")

    def test_reward_and_regret_populated(self):
        rec = self.history[N_ROUNDS - 1]
        self.assertIsNotNone(rec.reward)
        self.assertIsNotNone(rec.regret)

    def test_summary(self):
        s = self.runner.summary()
        self.assertEqual(s["rounds"], N_ROUNDS)
        self.assertGreater(s["cumulative_reward"], 0)


if __name__ == "__main__":
    unittest.main()

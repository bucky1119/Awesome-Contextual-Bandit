"""Step 3 自检 — 仿真统计 + Combined UCB 策略验证。

Done 标准：
  ✓ SimulationStats: T_s=0 时 radius_llm=+inf
  ✓ SimulationStats: 更新后 V(a) 正确
  ✓ SimulationStats: T_s(a) 单调递增
  ✓ SimulationStats: ucb_llm = mu_combined + radius_llm
  ✓ CombinedUCBPolicy: 冷启动 round-robin 正确
  ✓ CombinedUCBPolicy: score(a)=min(UCB^s1, UCB^s2, UCB^s3) 公式验证
  ✓ CombinedUCBPolicy: select 返回合法 arm_id
  ✓ CombinedUCBPolicy: debug 中有 ucb_s1/s2/s3
"""

import sys, os, unittest, math
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np

from OurMethod.core.protocol import Arm, Context, DecisionRecord, DsLlm
from OurMethod.core.sim_stats import SimulationStats
from OurMethod.core.combined_policy import CombinedUCBPolicy


K, Z_DIM = 4, 8


class TestSimulationStats(unittest.TestCase):
    def setUp(self):
        self.ss = SimulationStats(num_actions=K)

    def test_radius_inf_when_no_data(self):
        self.assertEqual(self.ss.radius_llm(0), float("inf"))

    def test_ucb_llm_inf_when_no_data(self):
        self.assertEqual(self.ss.ucb_llm(0), float("inf"))

    def test_ts_monotonic(self):
        ds = [DsLlm(arm_id=0, predicted_mean=0.5)]
        prev = self.ss.t_s[0]
        for _ in range(5):
            self.ss.update_from_ds_llm(ds)
            self.assertGreaterEqual(self.ss.t_s[0], prev)
            prev = self.ss.t_s[0]

    def test_v_changes(self):
        # 初始 V=0（两边都为0）
        self.assertAlmostEqual(self.ss.V(0), 0.0)
        # 给 LLM 一个均值
        self.ss.update_from_ds_llm([DsLlm(arm_id=0, predicted_mean=0.8)])
        # 给在线一个不同均值
        self.ss.update_online(0, 0.2)
        v = self.ss.V(0)
        self.assertGreater(v, 0.0, "V(a) 应为非零")

    def test_radius_finite_after_update(self):
        self.ss.update_from_ds_llm([DsLlm(arm_id=1, predicted_mean=0.5)])
        self.ss.update_online(1, 0.5)
        r = self.ss.radius_llm(1)
        self.assertTrue(math.isfinite(r))

    def test_ucb_llm_equals_mu_combined_plus_radius(self):
        """ucb_llm(a) = mu_combined(a) + radius_llm(a)"""
        self.ss.update_from_ds_llm([DsLlm(arm_id=0, predicted_mean=0.7)])
        self.ss.update_online(0, 0.3)
        ucb = self.ss.ucb_llm(0)
        expected = self.ss.mu_combined(0) + self.ss.radius_llm(0)
        self.assertAlmostEqual(ucb, expected, places=10)


class TestCombinedUCBPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = CombinedUCBPolicy(num_actions=K, z_dim=Z_DIM)
        self.arms = [Arm(arm_id=i) for i in range(K)]

    def _ctx(self) -> Context:
        return Context(features=np.random.randn(5))

    def _z(self) -> np.ndarray:
        v = np.random.randn(Z_DIM)
        return v / (np.linalg.norm(v) + 1e-8)

    def test_cold_start_round_robin(self):
        """冷启动阶段应按顺序遍历每个 arm。"""
        for i in range(K):
            a = self.policy.select(self._ctx(), self.arms, self._z())
            self.assertEqual(a, i, f"冷启动 step {i} 应选 arm {i}")
            rec = DecisionRecord(
                step=i, context=self._ctx(), arms=self.arms,
                prompt="", z_t=self._z(), chosen_arm=a, reward=0.5,
            )
            self.policy.update(rec)

    def test_select_returns_valid_arm(self):
        # 完成冷启动
        for i in range(K):
            z = self._z()
            a = self.policy.select(self._ctx(), self.arms, z)
            rec = DecisionRecord(
                step=i, context=self._ctx(), arms=self.arms,
                prompt="", z_t=z, chosen_arm=a, reward=np.random.rand(),
            )
            self.policy.update(rec)
        # 后续选择
        for _ in range(50):
            a = self.policy.select(self._ctx(), self.arms, self._z())
            self.assertIn(a, range(K))

    def test_min_of_three_ucb_formula(self):
        """验证 score(a) = min(UCB^s1, UCB^s2, UCB^s3)。"""
        np.random.seed(123)
        # 完成冷启动
        for i in range(K):
            z = self._z()
            a = self.policy.select(self._ctx(), self.arms, z)
            rec = DecisionRecord(
                step=i, context=self._ctx(), arms=self.arms,
                prompt="", z_t=z, chosen_arm=a, reward=0.5,
            )
            self.policy.update(rec)

        # 传入 ds_llm 更新仿真统计
        ds = [DsLlm(arm_id=i, predicted_mean=0.6) for i in range(K)]
        z = self._z()
        self.policy.select(self._ctx(), self.arms, z, ds_llm=ds)
        debug = self.policy.get_last_debug()

        # 验证 debug 中有 ucb_s1, ucb_s2, ucb_s3
        self.assertIsNotNone(debug["ucb_s1"])
        self.assertIsNotNone(debug["ucb_s2"])
        self.assertIsNotNone(debug["ucb_s3"])
        self.assertEqual(len(debug["ucb_s1"]), K)
        self.assertEqual(len(debug["ucb_s2"]), K)
        self.assertEqual(len(debug["ucb_s3"]), K)

        # 验证 ucb_values[a] == min(ucb_s1[a], ucb_s2[a], ucb_s3[a])
        for a in range(K):
            expected = min(debug["ucb_s1"][a], debug["ucb_s2"][a], debug["ucb_s3"][a])
            self.assertAlmostEqual(
                debug["ucb_values"][a], expected, places=10,
                msg=f"arm {a}: ucb_values 应为三个完整 UCB 取 min",
            )

    def test_ucb_s1_is_exploit_plus_linucb_radius(self):
        """验证 UCB^s1 = θ_a^T z + α * sqrt(z^T A^{-1} z)。"""
        np.random.seed(456)
        # 完成冷启动
        for i in range(K):
            z = self._z()
            a = self.policy.select(self._ctx(), self.arms, z)
            rec = DecisionRecord(
                step=i, context=self._ctx(), arms=self.arms,
                prompt="", z_t=z, chosen_arm=a, reward=0.8,
            )
            self.policy.update(rec)

        z = self._z()
        self.policy.select(self._ctx(), self.arms, z)
        debug = self.policy.get_last_debug()

        for a in range(K):
            theta_a = self.policy.get_theta(a)
            exploit = float(z @ theta_a)
            radius = self.policy.alpha_linucb * math.sqrt(
                float(z @ self.policy.A_inv[a] @ z)
            )
            expected_s1 = exploit + radius
            self.assertAlmostEqual(
                debug["ucb_s1"][a], expected_s1, places=8,
                msg=f"arm {a}: UCB^s1 应为 θ^T z + α√(z^T A^-1 z)",
            )

    def test_ucb_s2_is_mu_hat_plus_ucb1_radius(self):
        """验证 UCB^s2 = μ̂_a + c1 * sqrt(ln(t)/N_a)。"""
        np.random.seed(789)
        # 冷启动 + 若干轮
        for i in range(K * 2):
            z = self._z()
            a = self.policy.select(self._ctx(), self.arms, z)
            rec = DecisionRecord(
                step=i, context=self._ctx(), arms=self.arms,
                prompt="", z_t=z, chosen_arm=a, reward=np.random.rand(),
            )
            self.policy.update(rec)

        z = self._z()
        self.policy.select(self._ctx(), self.arms, z)
        debug = self.policy.get_last_debug()

        for a in range(K):
            if self.policy.counts[a] > 0:
                mu_hat = self.policy.sum_rewards[a] / self.policy.counts[a]
                radius = self.policy.c_ucb1 * math.sqrt(
                    math.log(self.policy.t + 1) / self.policy.counts[a]
                )
                expected_s2 = mu_hat + radius
                self.assertAlmostEqual(
                    debug["ucb_s2"][a], expected_s2, places=8,
                    msg=f"arm {a}: UCB^s2 应为 μ̂ + c√(ln(t)/N)",
                )

    def test_get_theta(self):
        theta = self.policy.get_theta(0)
        self.assertEqual(theta.shape, (Z_DIM,))


if __name__ == "__main__":
    unittest.main()

"""Step 0 自检 — 协议对象 + 四大接口 ABC 验证。

Done 标准：
  ✓ Context / Arm / DsLlm / DecisionRecord 可正常创建
  ✓ 三个协议对象 to_dict ↔ from_dict 往返一致
  ✓ DecisionRecord to_json ↔ from_json 往返一致
  ✓ 四大 ABC 存在且不能被直接实例化
"""

import sys, os, unittest, json
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np

from OurMethod.core.protocol import (
    Context, Arm, DsLlm, DecisionRecord,
    PromptBuilder, FrozenLLMEncoder, Compressor, Policy,
)


class TestContext(unittest.TestCase):
    def test_create_and_dim(self):
        c = Context(features=np.zeros(10), user_id="u1", machine_id="m1")
        self.assertEqual(c.dim(), 10)

    def test_round_trip(self):
        c = Context(
            features=np.random.randn(8),
            user_id="u42",
            machine_id="m7",
            env_fields={"temperature": 22.5},
            stat_features=np.array([1.0, 2.0]),
        )
        d = c.to_dict()
        c2 = Context.from_dict(d)
        np.testing.assert_allclose(c.features, c2.features, atol=1e-10)
        self.assertEqual(c.user_id, c2.user_id)
        np.testing.assert_allclose(c.stat_features, c2.stat_features, atol=1e-10)


class TestArm(unittest.TestCase):
    def test_create(self):
        a = Arm(arm_id=0, name="action_0", attributes={"cost": 1.5})
        self.assertEqual(a.arm_id, 0)

    def test_round_trip(self):
        a = Arm(arm_id=3, name="beta", description="test arm", attributes={"k": "v"})
        a2 = Arm.from_dict(a.to_dict())
        self.assertEqual(a.arm_id, a2.arm_id)
        self.assertEqual(a.name, a2.name)
        self.assertEqual(a.attributes, a2.attributes)


class TestDsLlm(unittest.TestCase):
    def test_create(self):
        d = DsLlm(arm_id=1, predicted_mean=0.6, predicted_std=0.1, reasoning="good")
        self.assertAlmostEqual(d.predicted_mean, 0.6)

    def test_round_trip(self):
        d = DsLlm(arm_id=2, predicted_mean=0.3, predicted_std=0.2, reasoning="ok")
        d2 = DsLlm.from_dict(d.to_dict())
        self.assertEqual(d.arm_id, d2.arm_id)
        self.assertAlmostEqual(d.predicted_mean, d2.predicted_mean)


class TestDecisionRecord(unittest.TestCase):
    def _make_record(self) -> DecisionRecord:
        ctx = Context(features=np.random.randn(5), user_id="u1")
        arms = [Arm(arm_id=i) for i in range(3)]
        ds = [DsLlm(arm_id=i, predicted_mean=0.5) for i in range(3)]
        return DecisionRecord(
            step=7,
            context=ctx,
            arms=arms,
            prompt="test prompt",
            h_t=np.random.randn(16),
            z_t=np.random.randn(8),
            ds_llm=ds,
            chosen_arm=1,
            reward=0.8,
            optimal_reward=1.0,
            ucb_values=[1.0, 1.2, 0.9],
            radius_ucb1=[0.5, 0.4, 0.6],
            radius_linucb=[0.3, 0.3, 0.3],
            radius_llm=[0.7, 0.8, 0.6],
            algorithm="combined_ucb",
        )

    def test_regret_auto_calc(self):
        rec = self._make_record()
        self.assertAlmostEqual(rec.regret, 0.2)

    def test_json_round_trip(self):
        rec = self._make_record()
        s = rec.to_json()
        rec2 = DecisionRecord.from_json(s)
        self.assertEqual(rec.step, rec2.step)
        self.assertEqual(rec.chosen_arm, rec2.chosen_arm)
        self.assertAlmostEqual(rec.reward, rec2.reward)
        np.testing.assert_allclose(rec.h_t, rec2.h_t, atol=1e-10)
        np.testing.assert_allclose(rec.z_t, rec2.z_t, atol=1e-10)
        self.assertEqual(len(rec2.ds_llm), 3)
        self.assertEqual(rec2.radius_ucb1, rec.radius_ucb1)

    def test_all_required_fields_in_dict(self):
        rec = self._make_record()
        d = rec.to_dict()
        for key in [
            "step", "context", "arms", "prompt", "h_t", "z_t", "ds_llm",
            "chosen_arm", "reward", "optimal_reward", "regret",
            "ucb_values", "radius_ucb1", "radius_linucb", "radius_llm",
            "algorithm", "timestamp",
        ]:
            self.assertIn(key, d, f"缺少字段: {key}")


class TestABCsNotInstantiable(unittest.TestCase):
    def test_prompt_builder_abc(self):
        with self.assertRaises(TypeError):
            PromptBuilder()

    def test_frozen_llm_abc(self):
        with self.assertRaises(TypeError):
            FrozenLLMEncoder()

    def test_compressor_abc(self):
        with self.assertRaises(TypeError):
            Compressor()

    def test_policy_abc(self):
        with self.assertRaises(TypeError):
            Policy()


if __name__ == "__main__":
    unittest.main()

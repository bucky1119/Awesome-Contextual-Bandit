"""Step 8 自检 — 真实冻结 LLM / JSON 解析 / 注册表 / 冷启动。

Done 标准：
  ✓ _parse_ds_llm_from_text: 正确 JSON 数组 / markdown 代码块 / 散落对象 / 垃圾回退
  ✓ FrozenLLMRegistry: register / get / get_default / list / 缺失报错
  ✓ StubFrozenLLM: generate_ds_llm 属性可切换
  ✓ ColdStartSimulator: 用 Stub LLM 预填充 sim_stats
  ✓ StructuredPromptBuilder.build_messages: 返回 chat 消息列表
  ✓ GenerativeFrozenLLM: 接口正确（跳过如无 transformers）
"""

import sys
import os
import unittest

_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np

from OurMethod.core.protocol import Arm, Context, DecisionRecord, DsLlm
from OurMethod.core.frozen_llm import (
    StubFrozenLLM, RealFrozenLLM, FrozenLLMRegistry,
    _parse_ds_llm_from_text, _HAS_TRANSFORMERS,
)
from OurMethod.core.prompt_builder import StructuredPromptBuilder
from OurMethod.core.sim_stats import SimulationStats
from OurMethod.core.cold_start import ColdStartSimulator


# ====================================================================== #
#  _parse_ds_llm_from_text                                               #
# ====================================================================== #

class TestParseDsLlm(unittest.TestCase):
    """JSON 解析鲁棒性测试。"""

    def test_valid_json_array(self):
        text = '[{"arm_id": 0, "predicted_mean": 0.8, "predicted_std": 0.1, "reasoning": "good"}, ' \
               '{"arm_id": 1, "predicted_mean": 0.3, "predicted_std": 0.2, "reasoning": "bad"}]'
        ds = _parse_ds_llm_from_text(text, 2)
        self.assertEqual(len(ds), 2)
        self.assertAlmostEqual(ds[0].predicted_mean, 0.8)
        self.assertAlmostEqual(ds[1].predicted_mean, 0.3)

    def test_markdown_code_block(self):
        text = 'Here is the result:\n```json\n[{"arm_id": 0, "predicted_mean": 0.5, "predicted_std": 0.3, "reasoning": "x"}]\n```'
        ds = _parse_ds_llm_from_text(text, 1)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0].arm_id, 0)

    def test_scattered_objects(self):
        text = 'For arm 0: {"arm_id": 0, "predicted_mean": 0.6, "predicted_std": 0.2}\n' \
               'For arm 1: {"arm_id": 1, "predicted_mean": 0.4, "predicted_std": 0.3}'
        ds = _parse_ds_llm_from_text(text, 2)
        self.assertEqual(len(ds), 2)
        self.assertAlmostEqual(ds[0].predicted_mean, 0.6)

    def test_garbage_fallback(self):
        text = "I have no idea what to output here. Random text."
        ds = _parse_ds_llm_from_text(text, 3)
        self.assertEqual(len(ds), 3)
        # 回退: 均匀预测
        for d in ds:
            self.assertAlmostEqual(d.predicted_mean, 1.0 / 3, places=4)
            self.assertEqual(d.predicted_std, 0.5)

    def test_empty_text(self):
        ds = _parse_ds_llm_from_text("", 2)
        self.assertEqual(len(ds), 2)

    def test_missing_arms_filled(self):
        """只包含部分 arm → 缺失的 arm 自动补上默认值。"""
        text = '[{"arm_id": 1, "predicted_mean": 0.9, "predicted_std": 0.1, "reasoning": "yes"}]'
        ds = _parse_ds_llm_from_text(text, 3)
        self.assertEqual(len(ds), 3)
        # arm 1 有实际值
        self.assertAlmostEqual(ds[1].predicted_mean, 0.9)
        # arm 0, 2 应为默认值
        self.assertAlmostEqual(ds[0].predicted_mean, 0.5)
        self.assertAlmostEqual(ds[2].predicted_mean, 0.5)

    def test_clip_values(self):
        """predicted_mean 超出 [0,1] 应被 clip。"""
        text = '[{"arm_id": 0, "predicted_mean": 2.5, "predicted_std": -0.5, "reasoning": ""}]'
        ds = _parse_ds_llm_from_text(text, 1)
        self.assertLessEqual(ds[0].predicted_mean, 1.0)
        self.assertGreater(ds[0].predicted_std, 0)

    def test_raw_text_preserved(self):
        text = '[{"arm_id": 0, "predicted_mean": 0.5, "predicted_std": 0.1, "reasoning": "ok"}]'
        ds = _parse_ds_llm_from_text(text, 1)
        self.assertEqual(ds[0].raw_text, text)


# ====================================================================== #
#  FrozenLLMRegistry                                                     #
# ====================================================================== #

class TestFrozenLLMRegistry(unittest.TestCase):

    def test_register_and_get(self):
        reg = FrozenLLMRegistry()
        stub = StubFrozenLLM(hidden_dim=32)
        reg.register("test_stub", stub)
        self.assertIs(reg.get("test_stub"), stub)

    def test_default(self):
        reg = FrozenLLMRegistry()
        s1 = StubFrozenLLM(hidden_dim=32)
        s2 = StubFrozenLLM(hidden_dim=64)
        reg.register("s1", s1)
        reg.register("s2", s2, default=True)
        self.assertIs(reg.get_default(), s2)
        self.assertEqual(reg.default_name, "s2")

    def test_first_registered_is_default(self):
        reg = FrozenLLMRegistry()
        s = StubFrozenLLM()
        reg.register("first", s)
        self.assertIs(reg.get_default(), s)

    def test_missing_raises(self):
        reg = FrozenLLMRegistry()
        with self.assertRaises(KeyError):
            reg.get("nonexistent")

    def test_empty_default_raises(self):
        reg = FrozenLLMRegistry()
        with self.assertRaises(RuntimeError):
            reg.get_default()

    def test_list_models(self):
        reg = FrozenLLMRegistry()
        reg.register("a", StubFrozenLLM())
        reg.register("b", StubFrozenLLM())
        self.assertEqual(sorted(reg.list_models()), ["a", "b"])

    def test_contains(self):
        reg = FrozenLLMRegistry()
        reg.register("x", StubFrozenLLM())
        self.assertIn("x", reg)
        self.assertNotIn("y", reg)

    def test_len(self):
        reg = FrozenLLMRegistry()
        self.assertEqual(len(reg), 0)
        reg.register("a", StubFrozenLLM())
        self.assertEqual(len(reg), 1)


# ====================================================================== #
#  StubFrozenLLM generate_ds_llm 属性                                    #
# ====================================================================== #

class TestStubToggle(unittest.TestCase):

    def test_toggle_ds_llm_off(self):
        llm = StubFrozenLLM(hidden_dim=32, generate_ds_llm=True)
        _, ds1 = llm.encode("test", num_arms=3)
        self.assertIsNotNone(ds1)

        llm.generate_ds_llm = False
        _, ds2 = llm.encode("test", num_arms=3)
        self.assertIsNone(ds2)

    def test_toggle_ds_llm_on(self):
        llm = StubFrozenLLM(hidden_dim=32, generate_ds_llm=False)
        _, ds1 = llm.encode("test", num_arms=3)
        self.assertIsNone(ds1)

        llm.generate_ds_llm = True
        _, ds2 = llm.encode("test", num_arms=3)
        self.assertIsNotNone(ds2)


# ====================================================================== #
#  ColdStartSimulator                                                    #
# ====================================================================== #

class TestColdStartSimulator(unittest.TestCase):

    def setUp(self):
        self.llm = StubFrozenLLM(hidden_dim=32, generate_ds_llm=True)
        self.pb = StructuredPromptBuilder()
        self.arms = [Arm(arm_id=i, name=f"arm_{i}") for i in range(3)]
        self.sim = ColdStartSimulator(self.llm, self.pb, self.arms)

    def test_simulate_returns_ds(self):
        contexts = [np.random.randn(5) for _ in range(10)]
        all_ds, all_h = self.sim.simulate(contexts)
        self.assertEqual(len(all_ds), 10)
        self.assertEqual(len(all_h), 10)
        for ds in all_ds:
            self.assertIsNotNone(ds)
            self.assertEqual(len(ds), 3)
        for h in all_h:
            self.assertEqual(h.shape[0], 32)  # hidden_dim=32

    def test_warmup_populates_sim_stats(self):
        from OurMethod.core.combined_policy import CombinedUCBPolicy
        policy = CombinedUCBPolicy(num_actions=3, z_dim=8)
        contexts = [np.random.randn(5) for _ in range(20)]

        # sim_stats 初始 T_s 全为 0
        for a in range(3):
            self.assertEqual(policy.sim_stats.t_s[a], 0)

        self.sim.warmup_policy(policy, contexts)

        # warmup 后 T_s > 0
        for a in range(3):
            self.assertGreater(policy.sim_stats.t_s[a], 0)

    def test_warmup_sim_stats_direct(self):
        ss = SimulationStats(num_actions=3)
        contexts = [np.random.randn(5) for _ in range(5)]
        self.sim.warmup_sim_stats(ss, contexts)
        for a in range(3):
            self.assertGreater(ss.t_s[a], 0)


# ====================================================================== #
#  StructuredPromptBuilder.build_messages                                #
# ====================================================================== #

class TestBuildMessages(unittest.TestCase):

    def test_returns_list_of_dicts(self):
        pb = StructuredPromptBuilder()
        ctx = Context(features=np.array([0.1, 0.2, 0.3]))
        arms = [Arm(arm_id=0), Arm(arm_id=1)]
        msgs = pb.build_messages(ctx, arms, system_prompt="You are helpful.")
        self.assertIsInstance(msgs, list)
        self.assertEqual(len(msgs), 2)  # system + user
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        # user content 应包含标准段
        self.assertIn("[ROLE]", msgs[1]["content"])
        self.assertIn("[TASK]", msgs[1]["content"])

    def test_no_system_prompt(self):
        pb = StructuredPromptBuilder()
        ctx = Context(features=np.array([0.5]))
        arms = [Arm(arm_id=0)]
        msgs = pb.build_messages(ctx, arms)
        self.assertEqual(len(msgs), 1)  # 只有 user
        self.assertEqual(msgs[0]["role"], "user")


# ====================================================================== #
#  GenerativeFrozenLLM (仅接口验证，跳过如无 transformers)                 #
# ====================================================================== #

class TestGenerativeFrozenLLMInterface(unittest.TestCase):

    @unittest.skipUnless(_HAS_TRANSFORMERS, "transformers not installed")
    def test_import_error_without_transformers(self):
        """如果 transformers 可用，GenerativeFrozenLLM 应可导入。"""
        from OurMethod.core.frozen_llm import GenerativeFrozenLLM
        # 只验证类存在，不实际加载模型（太慢）
        self.assertTrue(callable(GenerativeFrozenLLM))

    def test_import_exists(self):
        """不管 transformers 是否可用，import 不应报错。"""
        from OurMethod.core.frozen_llm import GenerativeFrozenLLM
        self.assertIsNotNone(GenerativeFrozenLLM)


if __name__ == "__main__":
    unittest.main()

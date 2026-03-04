"""Step 4 自检 — 冻结 LLM 编码器验证。

Done 标准：
  ✓ Stub 模式: 相同 prompt → 相同 h_t（确定性）
  ✓ Stub 模式: h_t 维度正确
  ✓ Stub 模式: h_t L2 范数 ≈ 1
  ✓ Stub 模式: ds_llm 结构正确（num_arms 个 DsLlm）
  ✓ Real 模式: 如果 transformers 不可用，自动回退到 Stub 并正常工作
  ✓ 不同 prompt → 不同 h_t
"""

import sys, os, unittest
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np

from OurMethod.core.protocol import DsLlm
from OurMethod.core.frozen_llm import StubFrozenLLM, RealFrozenLLM


H_DIM = 64
N_ARMS = 5


class TestStubFrozenLLM(unittest.TestCase):
    def setUp(self):
        self.llm = StubFrozenLLM(hidden_dim=H_DIM, generate_ds_llm=True)

    def test_deterministic(self):
        h1, _ = self.llm.encode("hello world", num_arms=N_ARMS)
        h2, _ = self.llm.encode("hello world", num_arms=N_ARMS)
        np.testing.assert_array_equal(h1, h2, "相同 prompt 应产生相同 h_t")

    def test_dimension(self):
        h, _ = self.llm.encode("test", num_arms=N_ARMS)
        self.assertEqual(h.shape, (H_DIM,))

    def test_l2_norm(self):
        h, _ = self.llm.encode("test", num_arms=N_ARMS)
        np.testing.assert_allclose(np.linalg.norm(h), 1.0, atol=1e-6)

    def test_ds_llm_structure(self):
        _, ds = self.llm.encode("test prompt", num_arms=N_ARMS)
        self.assertIsNotNone(ds)
        self.assertEqual(len(ds), N_ARMS)
        for i, d in enumerate(ds):
            self.assertIsInstance(d, DsLlm)
            self.assertEqual(d.arm_id, i)
            self.assertGreater(d.predicted_std, 0)

    def test_different_prompts_different_h(self):
        h1, _ = self.llm.encode("prompt A", num_arms=0)
        h2, _ = self.llm.encode("prompt B", num_arms=0)
        self.assertFalse(np.allclose(h1, h2), "不同 prompt 应产生不同 h_t")

    def test_no_ds_llm_when_zero_arms(self):
        _, ds = self.llm.encode("test", num_arms=0)
        self.assertIsNone(ds)


class TestRealFrozenLLMFallback(unittest.TestCase):
    def test_fallback_works(self):
        """不管 transformers 是否可用，RealFrozenLLM 都应能 encode。"""
        llm = RealFrozenLLM(generate_ds_llm=True)
        h, ds = llm.encode("fallback test", num_arms=3)
        self.assertEqual(h.shape[0], llm.get_hidden_dim())
        # 如果回退，ds 可能为 None 或有内容，不强制要求
        self.assertTrue(h.shape[0] > 0)

    def test_get_hidden_dim(self):
        llm = RealFrozenLLM()
        dim = llm.get_hidden_dim()
        self.assertGreater(dim, 0)


if __name__ == "__main__":
    unittest.main()

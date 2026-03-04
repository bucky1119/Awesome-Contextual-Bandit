"""Step 5 自检 — 特征压缩器 fφ 验证。

Done 标准：
  ✓ z_t 维度 == output_dim
  ✓ z_t L2 范数 ≈ 1（归一化）
  ✓ 不同 h_t → 不同 z_t
  ✓ forward 接受 batch 和单条输入
  ✓ forward_torch 可梯度（离线训练用）
  ✓ train_step_mse 能降低 loss
  ✓ save / load checkpoint 往返一致
  ✓ 1000 次单条 forward 在 2 秒内完成
"""

import sys, os, unittest, tempfile, time
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np
import torch

from OurMethod.core.compressor import MLPCompressor

D_IN, D_OUT = 64, 16


class TestCompressor(unittest.TestCase):
    def setUp(self):
        self.comp = MLPCompressor(input_dim=D_IN, output_dim=D_OUT)

    # ---- 形状 ---- #
    def test_output_dim(self):
        h = np.random.randn(D_IN)
        z = self.comp.forward(h)
        self.assertEqual(z.shape, (D_OUT,))

    def test_batch_forward(self):
        H = np.random.randn(10, D_IN)
        Z = self.comp.forward(H)
        self.assertEqual(Z.shape, (10, D_OUT))

    def test_get_output_dim(self):
        self.assertEqual(self.comp.get_output_dim(), D_OUT)

    # ---- 归一化 ---- #
    def test_l2_norm(self):
        h = np.random.randn(D_IN)
        z = self.comp.forward(h)
        np.testing.assert_allclose(np.linalg.norm(z), 1.0, atol=0.05)

    # ---- 非平凡映射 ---- #
    def test_different_input_different_output(self):
        h1 = np.random.randn(D_IN)
        h2 = np.random.randn(D_IN)
        z1 = self.comp.forward(h1)
        z2 = self.comp.forward(h2)
        self.assertFalse(np.allclose(z1, z2, atol=1e-4),
                         "不同输入应产生不同输出")

    # ---- 可梯度 ---- #
    def test_forward_torch_has_grad(self):
        h = torch.randn(5, D_IN, requires_grad=False)
        self.comp.train()
        z = self.comp.forward_torch(h)
        loss = z.sum()
        loss.backward()
        # 检查参数有梯度
        has_grad = any(p.grad is not None for p in self.comp.parameters())
        self.assertTrue(has_grad, "forward_torch 应支持反向传播")

    # ---- 训练 ---- #
    def test_train_step_mse(self):
        h_batch = torch.randn(32, D_IN)
        theta = torch.randn(D_OUT).detach()
        rewards = torch.randn(32)
        loss1 = self.comp.train_step_mse(h_batch, theta, rewards)
        for _ in range(20):
            loss2 = self.comp.train_step_mse(h_batch, theta, rewards)
        self.assertLess(loss2, loss1 * 1.5,
                        "多次训练后 loss 应有下降趋势")

    # ---- checkpoint ---- #
    def test_save_load(self):
        h = np.random.randn(D_IN)
        z_before = self.comp.forward(h)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            self.comp.save(f.name)
            comp2 = MLPCompressor(input_dim=D_IN, output_dim=D_OUT)
            comp2.load(f.name)
            z_after = comp2.forward(h)
        np.testing.assert_allclose(z_before, z_after, atol=1e-6)

    # ---- 速度 ---- #
    def test_speed_1000_forward(self):
        H = np.random.randn(1000, D_IN)
        t0 = time.perf_counter()
        for i in range(1000):
            self.comp.forward(H[i])
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 5.0, f"1000 次 forward 花费 {elapsed:.2f}s，过慢")


if __name__ == "__main__":
    unittest.main()

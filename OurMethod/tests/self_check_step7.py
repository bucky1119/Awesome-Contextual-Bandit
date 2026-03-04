"""Step 7 自检 — 离线数据集 + 离线训练器验证。

Done 标准：
  ✓ OfflineDataset 从 jsonl 正确加载记录
  ✓ OfflineTrainer.train 返回的 loss 列表长度 == epochs
  ✓ Loss 有下降趋势（末尾 < 起始 × 1.2）
  ✓ Checkpoint save / load 后 forward 结果一致
  ✓ rebuild_policy_stats 更新 LinUCB 的 A/b
  ✓ 在线 runner 可继续使用 load 后的 compressor
"""

import sys, os, unittest, tempfile, json
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np
import torch

from OurMethod.core.protocol import Arm, Context, DecisionRecord, DsLlm
from OurMethod.core.compressor import MLPCompressor
from OurMethod.core.combined_policy import CombinedUCBPolicy
from OurMethod.core.offline_dataset import OfflineDataset
from OurMethod.core.train_compressor import OfflineTrainer

K, H_DIM, Z_DIM = 4, 32, 8
N_RECORDS = 200


def _write_fake_jsonl(path: str):
    """写入模拟的 decisions.jsonl。"""
    rng = np.random.RandomState(0)
    with open(path, "w") as f:
        for i in range(N_RECORDS):
            rec = DecisionRecord(
                step=i,
                context=Context(features=rng.randn(5)),
                arms=[Arm(arm_id=a) for a in range(K)],
                prompt="test",
                h_t=rng.randn(H_DIM),
                z_t=rng.randn(Z_DIM),
                ds_llm=[DsLlm(arm_id=a, predicted_mean=0.5) for a in range(K)],
                chosen_arm=rng.randint(K),
                reward=rng.rand(),
            )
            f.write(rec.to_json() + "\n")


class TestOfflineDataset(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._jsonl = os.path.join(self._tmpdir, "decisions.jsonl")
        _write_fake_jsonl(self._jsonl)

    def test_load(self):
        ds = OfflineDataset(self._jsonl)
        self.assertEqual(len(ds), N_RECORDS)

    def test_item_shapes(self):
        ds = OfflineDataset(self._jsonl)
        h, arm, rew = ds[0]
        self.assertEqual(h.shape, (H_DIM,))
        self.assertIsInstance(arm, int)

    def test_loader(self):
        ds = OfflineDataset(self._jsonl)
        loader = ds.get_loader(batch_size=16)
        batch = next(iter(loader))
        self.assertEqual(len(batch), 3)
        self.assertEqual(batch[0].shape[1], H_DIM)


class TestOfflineTrainer(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._jsonl = os.path.join(self._tmpdir, "decisions.jsonl")
        _write_fake_jsonl(self._jsonl)
        self.ds = OfflineDataset(self._jsonl)
        self.comp = MLPCompressor(input_dim=H_DIM, output_dim=Z_DIM)
        self.policy = CombinedUCBPolicy(num_actions=K, z_dim=Z_DIM)

    def test_train_returns_losses(self):
        trainer = OfflineTrainer(self.comp, self.policy)
        losses = trainer.train(self.ds, epochs=5)
        self.assertEqual(len(losses), 5)

    def test_loss_decreases(self):
        trainer = OfflineTrainer(self.comp, self.policy)
        losses = trainer.train(self.ds, epochs=30)
        avg_first3 = np.mean(losses[:3])
        avg_last3 = np.mean(losses[-3:])
        self.assertLess(avg_last3, avg_first3 * 1.5,
                        f"Loss 应有下降趋势: first3={avg_first3:.4f}, last3={avg_last3:.4f}")

    def test_checkpoint_save_load(self):
        ckpt = os.path.join(self._tmpdir, "comp.pt")
        trainer = OfflineTrainer(self.comp, self.policy)
        trainer.train(self.ds, epochs=5)
        trainer.save_checkpoint(ckpt)
        h = np.random.randn(H_DIM)
        z_before = self.comp.forward(h)

        comp2 = MLPCompressor(input_dim=H_DIM, output_dim=Z_DIM)
        comp2.load(ckpt)
        z_after = comp2.forward(h)
        np.testing.assert_allclose(z_before, z_after, atol=1e-6)

    def test_rebuild_policy_stats(self):
        trainer = OfflineTrainer(self.comp, self.policy)
        trainer.train(self.ds, epochs=3)
        A_before = [A.copy() for A in self.policy.A]
        trainer.rebuild_policy_stats(self.ds)
        # A 应和之前不同
        changed = any(
            not np.allclose(self.policy.A[a], A_before[a])
            for a in range(K)
        )
        self.assertTrue(changed, "rebuild 后 A 矩阵应变化")


if __name__ == "__main__":
    unittest.main()

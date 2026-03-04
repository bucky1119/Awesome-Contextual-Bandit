"""Step 7 — 离线训练器：只更新 fφ（θ 冻结）。

loss = MSE( θ_a^T fφ(h_t) , r )

θ_a 来自 CombinedUCBPolicy 的当前参数（离线快照），训练时 detach。
fφ = MLPCompressor，正常反向传播更新。
"""

from __future__ import annotations

import os
import sys
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

from typing import Dict, List, Optional

import numpy as np
import torch

from OurMethod.core.compressor import MLPCompressor
from OurMethod.core.combined_policy import CombinedUCBPolicy
from OurMethod.core.offline_dataset import OfflineDataset


class OfflineTrainer:
    """离线训练 fφ 的训练器。"""

    def __init__(
        self,
        compressor: MLPCompressor,
        policy: CombinedUCBPolicy,
        lr: float = 1e-3,
    ):
        # ============================================================ #
        #  参数指南 (Parameter Guide)
        # ============================================================ #
        #
        # 🟡 可按需调整
        #
        #   lr (float, default=1e-3)
        #       离线训练 fφ 的 Adam 学习率。
        #       会覆盖 MLPCompressor 自带的 optimizer。
        #       增大 → 训练更快但可能不稳定；减小 → 更稳定但收敛慢。
        #       典型范围: [1e-4, 5e-3]。
        #
        #   epochs (int, default=10)  [在 train() 中]
        #       离线训练遍历轮次。数据少时可增大，数据充足时 5~10 即可。
        #
        #   batch_size (int, default=64)  [在 train() 中]
        #       离线训练批大小。数据量小(< 500) 可减至 16~32。
        # ============================================================ #
        self.comp = compressor
        self.policy = policy
        # 重置或使用已有 optimizer
        self.comp.optimizer = torch.optim.Adam(self.comp.parameters(), lr=lr)

    def train(
        self,
        dataset: OfflineDataset,
        epochs: int = 10,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> List[float]:
        """训练 fφ 并返回每 epoch 的平均 loss。"""
        if len(dataset) == 0:
            return []

        loader = dataset.get_loader(batch_size=batch_size, shuffle=True)
        epoch_losses: List[float] = []

        for ep in range(epochs):
            total_loss = 0.0
            n_batches = 0
            self.comp.train()

            for h_batch, arm_batch, reward_batch in loader:
                self.comp.optimizer.zero_grad()
                z = self.comp.forward_torch(h_batch)          # (B, d_z)

                # 为每个样本获取对应 arm 的 θ_a（冻结）
                preds = []
                for i in range(len(h_batch)):
                    a = int(arm_batch[i])
                    theta_a = self.policy.get_theta(a)         # numpy
                    theta_t = torch.tensor(theta_a, dtype=torch.float32)  # detached
                    preds.append(z[i] @ theta_t)

                pred = torch.stack(preds)
                target = reward_batch.float()
                loss = torch.nn.functional.mse_loss(pred, target)
                loss.backward()
                self.comp.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg = total_loss / max(n_batches, 1)
            epoch_losses.append(avg)
            if verbose:
                print(f"  Epoch {ep+1}/{epochs}: loss={avg:.6f}")

        self.comp.eval()
        return epoch_losses

    def rebuild_policy_stats(self, dataset: OfflineDataset):
        """用新 fφ 重计算 z → 重建 LinUCB 的 A/b。"""
        if len(dataset) == 0:
            return
        all_h = torch.tensor(dataset._h, dtype=torch.float32)
        self.comp.eval()
        with torch.no_grad():
            all_z = self.comp.forward_torch(all_h).numpy().astype(np.float64)
        self.policy.rebuild_linucb(all_z, dataset._arms, dataset._rewards)

    def save_checkpoint(self, path: str):
        self.comp.save(path)

    def load_checkpoint(self, path: str):
        self.comp.load(path)

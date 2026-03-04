"""Step 5 — 特征压缩器 fφ : h_t → z_t。

轻量 MLP + L2 归一化；支持可选 extra 向量拼接。
在线阶段 fφ 固定（只 forward），离线阶段可训练（train_step）。
"""

from __future__ import annotations

import sys, os
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from OurMethod.core.protocol import Compressor as CompressorABC


class MLPCompressor(CompressorABC, nn.Module):
    """2 层 MLP 压缩器: h_t (d_in) -> z_t (d_out)，L2 归一化。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 32,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        lr: float = 1e-3,
    ):
        # ============================================================ #
        #  参数指南 (Parameter Guide)
        # ============================================================ #
        #
        # 🔴 重点关注
        #
        #   output_dim (int, default=32)
        #       即 z_dim — 压缩后特征维度，传递给 CombinedUCBPolicy。
        #       必须与 policy 的 z_dim 一致。
        #       典型范围: [8, 64]，建议 16~32。
        #
        # 🟡 可按需调整
        #
        #   hidden_dim (int, default=auto)
        #       MLP 中间层维度。默认 = max((input+output)//2, output*2)。
        #       增大 → 表达能力更强但过拟合风险高；通常默认即可。
        #
        #   dropout (float, default=0.1)
        #       训练时的 Dropout 率，防止离线训练过拟合。
        #       典型范围: [0.0, 0.3]。
        #
        #   lr (float, default=1e-3)
        #       在线结构保持训练的 Adam 学习率。
        #       离线训练时可被 OfflineTrainer 覆盖。
        #
        # 🟢 无需调整
        #
        #   input_dim (int) — 由冻结 LLM 的 hidden_dim 自动决定
        # ============================================================ #
        nn.Module.__init__(self)
        if hidden_dim is None:
            hidden_dim = max((input_dim + output_dim) // 2, output_dim * 2)
        self._in = input_dim
        self._out = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    # ---- CompressorABC 接口 ---- #
    def forward(
        self, h_t: np.ndarray, extra: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """numpy in → numpy out（eval 模式，无梯度）。"""
        vec = h_t
        if extra is not None:
            vec = np.concatenate([h_t, extra])
        t = torch.tensor(vec, dtype=torch.float32)
        squeeze = t.dim() == 1
        if squeeze:
            t = t.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            z = self.net(t)
            z = torch.nn.functional.normalize(z, p=2, dim=-1)
        z_np = z.numpy().astype(np.float64)
        return z_np.squeeze(0) if squeeze else z_np

    def get_output_dim(self) -> int:
        return self._out

    # ---- PyTorch 前向（可梯度，离线训练用） ---- #
    def forward_torch(
        self, h: torch.Tensor,
    ) -> torch.Tensor:
        z = self.net(h)
        return torch.nn.functional.normalize(z, p=2, dim=-1)

    # ---- 离线训练接口 ---- #
    def train_step_mse(
        self,
        h_batch: torch.Tensor,
        theta_frozen: torch.Tensor,
        rewards: torch.Tensor,
    ) -> float:
        """MSE 训练 fφ（θ 冻结）:  loss = (θ^T fφ(h) - r)^2。

        Args:
            h_batch: (B, d_in)
            theta_frozen: (d_out,)  —— 已 detach
            rewards: (B,)
        Returns:
            loss value
        """
        self.train()
        self.optimizer.zero_grad()
        z = self.forward_torch(h_batch)            # (B, d_out)
        pred = z @ theta_frozen                     # (B,)
        loss = torch.nn.functional.mse_loss(pred, rewards)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_step_structure(self, h_batch: torch.Tensor) -> float:
        """自监督结构保持训练（保持样本间距离）。"""
        self.train()
        self.optimizer.zero_grad()
        z = self.forward_torch(h_batch)
        h_norm = torch.nn.functional.normalize(h_batch, p=2, dim=-1)
        sim_h = h_norm @ h_norm.t()
        sim_z = z @ z.t()
        loss = torch.nn.functional.mse_loss(sim_z, sim_h)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # ---- checkpoint ---- #
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()

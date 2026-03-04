"""Step 7 — 离线数据集：从 decisions.jsonl 构建训练数据。"""

from __future__ import annotations

import json
import os
import sys
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from OurMethod.core.protocol import DecisionRecord


class OfflineDataset(Dataset):
    """从 decisions.jsonl 提取 (h_t, arm, reward) 用于离线训练 fφ。"""

    def __init__(self, jsonl_path: str):
        self.h_list: List[np.ndarray] = []
        self.arm_list: List[int] = []
        self.reward_list: List[float] = []

        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = DecisionRecord.from_json(line)
                if rec.h_t is not None:
                    self.h_list.append(rec.h_t)
                    self.arm_list.append(rec.chosen_arm)
                    self.reward_list.append(rec.reward)

        self._h = np.array(self.h_list, dtype=np.float32) if self.h_list else np.empty((0, 1), dtype=np.float32)
        self._arms = np.array(self.arm_list, dtype=np.int64)
        self._rewards = np.array(self.reward_list, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._h)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        return (
            torch.tensor(self._h[idx], dtype=torch.float32),
            int(self._arms[idx]),
            float(self._rewards[idx]),
        )

    def get_loader(self, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


class InMemoryOfflineDataset(Dataset):
    """内存构建的离线数据集，支持增量添加。

    用于：
      - 冷启动仿真数据 (h_t, arm_id, predicted_mean)
      - 在线阶段积累的真实数据 (h_t, arm_id, real_reward)

    与 OfflineTrainer 完全兼容（提供 _h, _arms, _rewards 属性）。
    """

    def __init__(self, h_dim: int):
        self.h_dim = h_dim
        self._h_list: List[np.ndarray] = []
        self._arm_list: List[int] = []
        self._reward_list: List[float] = []
        # 缓存 numpy 数组
        self._h_arr: Optional[np.ndarray] = None
        self._arms_arr: Optional[np.ndarray] = None
        self._rewards_arr: Optional[np.ndarray] = None
        self._dirty = True

    def add(self, h_t: np.ndarray, arm: int, reward: float):
        """添加单条样本 (h_t, arm_id, reward)。"""
        self._h_list.append(np.asarray(h_t, dtype=np.float32))
        self._arm_list.append(int(arm))
        self._reward_list.append(float(reward))
        self._dirty = True

    def _sync(self):
        """惰性重建 numpy 缓存。"""
        if self._dirty:
            if self._h_list:
                self._h_arr = np.array(self._h_list, dtype=np.float32)
                self._arms_arr = np.array(self._arm_list, dtype=np.int64)
                self._rewards_arr = np.array(self._reward_list, dtype=np.float32)
            else:
                self._h_arr = np.empty((0, self.h_dim), dtype=np.float32)
                self._arms_arr = np.empty(0, dtype=np.int64)
                self._rewards_arr = np.empty(0, dtype=np.float32)
            self._dirty = False

    @property
    def _h(self) -> np.ndarray:
        self._sync()
        return self._h_arr  # type: ignore

    @property
    def _arms(self) -> np.ndarray:
        self._sync()
        return self._arms_arr  # type: ignore

    @property
    def _rewards(self) -> np.ndarray:
        self._sync()
        return self._rewards_arr  # type: ignore

    def __len__(self) -> int:
        return len(self._h_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        return (
            torch.tensor(self._h_list[idx], dtype=torch.float32),
            int(self._arm_list[idx]),
            float(self._reward_list[idx]),
        )

    def get_loader(self, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

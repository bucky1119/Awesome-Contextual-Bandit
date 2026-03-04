"""实验检查点管理器：保存/恢复完整实验状态。

管理的数据：
  1. Compressor fφ 权重 (state_dict)
  2. Policy 状态 (LinUCB A/b/A_inv, UCB1 counts/sum_rewards, SimStats)
  3. 离线训练 loss 历史
  4. 每步 UCB 分项值 (哪个 UCB 主导)
  5. 完整实验配置

目录结构::

    OurMethod/checkpoints/
      {experiment_tag}/
        config.json              — 完整实验配置
        compressor_warmup.pt     — 热启动后的 fφ
        compressor_step{N}.pt    — 离线训练后的 fφ (每次离线训练一个)
        compressor_final.pt      — 最终 fφ
        policy_state.npz         — Policy 状态快照
        offline_losses.json      — 离线训练 loss 历史
        ucb_trace.csv            — 每步 UCB 分项值
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)


def _default_ckpt_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "checkpoints")


class CheckpointManager:
    """实验检查点管理器。

    负责在实验运行过程中：
    - 保存 Compressor fφ 权重
    - 保存 Policy 完整状态
    - 记录离线训练 loss 曲线
    - 追踪每步 UCB 分项值

    Args:
        experiment_tag: 实验标识 (用作目录名)
        base_dir:       检查点根目录，默认 OurMethod/checkpoints/
    """

    def __init__(
        self,
        experiment_tag: str,
        base_dir: Optional[str] = None,
    ):
        self.tag = experiment_tag
        self.base = base_dir or _default_ckpt_dir()
        self.exp_dir = os.path.join(self.base, experiment_tag)
        os.makedirs(self.exp_dir, exist_ok=True)

        # 离线训练 loss 历史: [{phase, step, epochs, losses}]
        self._offline_losses: List[Dict[str, Any]] = []

        # UCB trace buffer (批量写入)
        self._ucb_buffer: List[Dict[str, Any]] = []
        self._ucb_flush_every = 500

    # ================================================================== #
    #  1. 实验配置                                                        #
    # ================================================================== #
    def save_config(self, config: Dict[str, Any]):
        """保存完整实验配置。"""
        config["saved_at"] = datetime.now().isoformat()
        path = os.path.join(self.exp_dir, "config.json")
        with open(path, "w") as f:
            json.dump(config, f, ensure_ascii=False, indent=2, default=str)
        return path

    def load_config(self) -> Dict[str, Any]:
        path = os.path.join(self.exp_dir, "config.json")
        with open(path, "r") as f:
            return json.load(f)

    # ================================================================== #
    #  2. Compressor fφ 权重                                              #
    # ================================================================== #
    def save_compressor(self, compressor, label: str = "final"):
        """保存 Compressor 权重。

        label: "warmup" | "step500" | "step1000" | "final" 等
        """
        import torch
        path = os.path.join(self.exp_dir, f"compressor_{label}.pt")
        torch.save(compressor.state_dict(), path)
        return path

    def load_compressor(self, compressor, label: str = "final"):
        """加载 Compressor 权重。"""
        import torch
        path = os.path.join(self.exp_dir, f"compressor_{label}.pt")
        compressor.load_state_dict(torch.load(path, map_location="cpu"))
        compressor.eval()

    def list_compressor_checkpoints(self) -> List[str]:
        """列出所有可用的 compressor 检查点标签。"""
        labels = []
        for f in os.listdir(self.exp_dir):
            if f.startswith("compressor_") and f.endswith(".pt"):
                label = f[len("compressor_"):-len(".pt")]
                labels.append(label)
        return sorted(labels)

    # ================================================================== #
    #  3. Policy 完整状态                                                  #
    # ================================================================== #
    def save_policy_state(self, policy, label: str = "final"):
        """保存 CombinedUCBPolicy 完整状态 (LinUCB + UCB1 + SimStats)。"""
        from OurMethod.core.combined_policy import CombinedUCBPolicy

        data = {
            # UCB1
            "counts": policy.counts,
            "sum_rewards": policy.sum_rewards,
            "t": np.array([policy.t]),
            # LinUCB
            "z_dim": np.array([policy.z_dim]),
            "lambda_reg": np.array([policy.lambda_reg]),
        }

        # LinUCB A, b, A_inv (num_actions 个矩阵)
        for a in range(policy.K):
            data[f"A_{a}"] = policy.A[a]
            data[f"b_{a}"] = policy.b_vec[a]
            data[f"A_inv_{a}"] = policy.A_inv[a]

        # SimStats
        ss = policy.sim_stats
        data["ss_sum_online"] = ss.sum_online
        data["ss_n_online"] = ss.n_online
        data["ss_sum_llm"] = ss.sum_llm
        data["ss_t_s"] = ss.t_s
        data["ss_total_t"] = np.array([ss.total_t])
        data["ss_c_sim"] = np.array([ss.c_sim])

        path = os.path.join(self.exp_dir, f"policy_state_{label}.npz")
        np.savez(path, **data)
        return path

    def load_policy_state(self, policy, label: str = "final"):
        """恢复 CombinedUCBPolicy 完整状态。"""
        path = os.path.join(self.exp_dir, f"policy_state_{label}.npz")
        data = np.load(path)

        # UCB1
        policy.counts[:] = data["counts"]
        policy.sum_rewards[:] = data["sum_rewards"]
        policy.t = int(data["t"][0])

        # LinUCB
        for a in range(policy.K):
            policy.A[a] = data[f"A_{a}"]
            policy.b_vec[a] = data[f"b_{a}"]
            policy.A_inv[a] = data[f"A_inv_{a}"]

        # SimStats
        ss = policy.sim_stats
        ss.sum_online[:] = data["ss_sum_online"]
        ss.n_online[:] = data["ss_n_online"]
        ss.sum_llm[:] = data["ss_sum_llm"]
        ss.t_s[:] = data["ss_t_s"]
        ss.total_t = int(data["ss_total_t"][0])
        ss.c_sim = float(data["ss_c_sim"][0])

    # ================================================================== #
    #  4. 离线训练 loss 历史                                               #
    # ================================================================== #
    def record_offline_loss(
        self,
        phase: str,
        step: int,
        losses: List[float],
        data_size: int = 0,
    ):
        """记录一次离线训练的 loss 列表。

        Args:
            phase:     "warmup" | "periodic"
            step:      离线训练触发时的在线步数 (-1 表示 warmup)
            losses:    每 epoch 的平均 loss
            data_size: 训练数据条数
        """
        self._offline_losses.append({
            "phase": phase,
            "trigger_step": step,
            "data_size": data_size,
            "epochs": len(losses),
            "losses": [round(l, 8) for l in losses],
            "final_loss": round(losses[-1], 8) if losses else None,
        })

    def save_offline_losses(self):
        """写入磁盘。"""
        path = os.path.join(self.exp_dir, "offline_losses.json")
        with open(path, "w") as f:
            json.dump(self._offline_losses, f, indent=2, ensure_ascii=False)
        return path

    def load_offline_losses(self) -> List[Dict[str, Any]]:
        path = os.path.join(self.exp_dir, "offline_losses.json")
        with open(path, "r") as f:
            return json.load(f)

    # ================================================================== #
    #  5. 每步 UCB 分项追踪                                                #
    # ================================================================== #
    def record_ucb_trace(
        self,
        step: int,
        chosen_arm: int,
        reward: float,
        ucb_s1: List[float],
        ucb_s2: List[float],
        ucb_s3: List[float],
        min_ucb: List[float],
    ):
        """记录一步的 UCB 分项值。"""
        binding = []
        for a in range(len(min_ucb)):
            vals = [ucb_s1[a], ucb_s2[a], ucb_s3[a]]
            binding.append(["s1", "s2", "s3"][int(np.argmin(vals))])

        row = {
            "step": step,
            "chosen_arm": chosen_arm,
            "reward": reward,
        }
        for a in range(len(min_ucb)):
            row[f"ucb_s1_a{a}"] = round(ucb_s1[a], 6) if ucb_s1[a] != float("inf") else "inf"
            row[f"ucb_s2_a{a}"] = round(ucb_s2[a], 6) if ucb_s2[a] != float("inf") else "inf"
            row[f"ucb_s3_a{a}"] = round(ucb_s3[a], 6) if ucb_s3[a] != float("inf") else "inf"
            row[f"binding_a{a}"] = binding[a]
        self._ucb_buffer.append(row)

        if len(self._ucb_buffer) >= self._ucb_flush_every:
            self._flush_ucb_trace()

    def _flush_ucb_trace(self):
        """批量写入 UCB trace 到 CSV。"""
        if not self._ucb_buffer:
            return
        import pandas as pd
        path = os.path.join(self.exp_dir, "ucb_trace.csv")
        df = pd.DataFrame(self._ucb_buffer)
        header = not os.path.exists(path)
        df.to_csv(path, mode="a", header=header, index=False)
        self._ucb_buffer.clear()

    def finalize_ucb_trace(self):
        """实验结束时刷入剩余数据。"""
        self._flush_ucb_trace()

    # ================================================================== #
    #  6. UCB 主导统计                                                     #
    # ================================================================== #
    def compute_binding_stats(self) -> Dict[str, Any]:
        """分析哪个 UCB 在各阶段主导决策。"""
        import pandas as pd
        path = os.path.join(self.exp_dir, "ucb_trace.csv")
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path)
        binding_cols = [c for c in df.columns if c.startswith("binding_")]
        if not binding_cols:
            return {}

        # 对 chosen_arm 统计 binding
        chosen_bindings = []
        for _, row in df.iterrows():
            a = int(row["chosen_arm"])
            col = f"binding_a{a}"
            if col in row:
                chosen_bindings.append(row[col])
        if not chosen_bindings:
            return {}

        from collections import Counter
        total = len(chosen_bindings)
        counts = Counter(chosen_bindings)
        stats = {
            "total_steps": total,
            "s1_dominant": counts.get("s1", 0),
            "s2_dominant": counts.get("s2", 0),
            "s3_dominant": counts.get("s3", 0),
            "s1_pct": round(counts.get("s1", 0) / total * 100, 1),
            "s2_pct": round(counts.get("s2", 0) / total * 100, 1),
            "s3_pct": round(counts.get("s3", 0) / total * 100, 1),
        }

        # 分段统计
        n_segs = 5
        seg_size = max(total // n_segs, 1)
        for s_idx in range(n_segs):
            lo = s_idx * seg_size
            hi = min((s_idx + 1) * seg_size, total)
            seg = chosen_bindings[lo:hi]
            seg_counts = Counter(seg)
            seg_total = len(seg)
            if seg_total > 0:
                stats[f"seg{s_idx}_range"] = f"[{lo},{hi})"
                stats[f"seg{s_idx}_s1_pct"] = round(
                    seg_counts.get("s1", 0) / seg_total * 100, 1)
                stats[f"seg{s_idx}_s2_pct"] = round(
                    seg_counts.get("s2", 0) / seg_total * 100, 1)
                stats[f"seg{s_idx}_s3_pct"] = round(
                    seg_counts.get("s3", 0) / seg_total * 100, 1)
        return stats

    # ================================================================== #
    #  7. 汇总                                                            #
    # ================================================================== #
    def save_experiment_summary(self, extra: Optional[Dict[str, Any]] = None):
        """保存完整实验汇总 (config + losses + binding stats)。"""
        summary: Dict[str, Any] = {"experiment_tag": self.tag}

        # config
        cfg_path = os.path.join(self.exp_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                summary["config"] = json.load(f)

        # offline losses summary
        if self._offline_losses:
            summary["offline_training"] = {
                "total_phases": len(self._offline_losses),
                "phases": self._offline_losses,
            }

        # binding stats
        binding = self.compute_binding_stats()
        if binding:
            summary["ucb_binding_stats"] = binding

        # compressor checkpoints
        ckpts = self.list_compressor_checkpoints()
        summary["compressor_checkpoints"] = ckpts

        if extra:
            summary.update(extra)

        summary["finalized_at"] = datetime.now().isoformat()
        path = os.path.join(self.exp_dir, "experiment_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        return path

    # ================================================================== #
    #  工具                                                               #
    # ================================================================== #
    @property
    def dir(self) -> str:
        return self.exp_dir

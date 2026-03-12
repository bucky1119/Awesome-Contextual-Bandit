"""冷启动仿真缓存：一次生成，多次复用。

对于一个特定数据集+模型组合，冷启动仿真只需运行一次。缓存的核心数据包括：
  - all_h:     每个上下文的 LLM hidden state (h_t)
  - ds_llm:    每个上下文的 LLM 预测 (arm_id, predicted_mean)
  - sim_stats: 由 ds_llm 预填充后的 SimulationStats 快照
  - metadata:  模型名、维度、生成时间等可追溯信息

缓存目录结构::

    OurMethod/
      cold_start_data/
        statlog_smollm2_n50_seed42/
          metadata.json       — 可追溯元信息
          all_h.npy           — (cold_start_n, h_dim) float32
          ds_llm.json         — [[{arm_id, predicted_mean, confidence}, ...], ...]
          sim_stats.npz       — sum_llm, t_s 数组
          contexts.npy        — (cold_start_n, context_dim) 用于验证数据一致性

用法::

    cache = ColdStartCache(base_dir="OurMethod/cold_start_data")

    # 生成并保存
    cache.save(tag="statlog_smollm2_n50_seed42",
               all_h=all_h, all_ds=all_ds,
               sim_stats=policy.sim_stats,
               contexts=cs_contexts, metadata={...})

    # 之后的实验直接加载
    data = cache.load("statlog_smollm2_n50_seed42")
    all_h = data["all_h"]
    all_ds = data["all_ds"]
    # → 跳过冷启动，直接进入热启动训练 / 在线决策

    TODO:命名应该改为仿真数据条数，而不是n的轮次
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

from OurMethod.core.protocol import DsLlm
from OurMethod.core.sim_stats import SimulationStats


def _default_base_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "cold_start_data")


class ColdStartCache:
    """冷启动仿真数据缓存管理器。

    Args:
        base_dir: 缓存根目录，默认 OurMethod/cold_start_data/
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or _default_base_dir()
        os.makedirs(self.base_dir, exist_ok=True)

    # ---------------------------------------------------------------- #
    #  Tag 命名约定                                                     #
    # ---------------------------------------------------------------- #
    @staticmethod
    def make_tag(
        dataset: str,
        model: str,
        cold_start_n: int,
        seed: int,
    ) -> str:
        """生成缓存标签。

        Examples:
            "statlog_smollm2_n50_seed42"
            "mushroom_stub_n100_seed0"
        """
        return f"{dataset}_{model}_n{cold_start_n}_seed{seed}"

    def tag_dir(self, tag: str) -> str:
        return os.path.join(self.base_dir, tag)

    def exists(self, tag: str) -> bool:
        d = self.tag_dir(tag)
        return (
            os.path.isdir(d)
            and os.path.isfile(os.path.join(d, "metadata.json"))
            and os.path.isfile(os.path.join(d, "all_h.npy"))
            and os.path.isfile(os.path.join(d, "ds_llm.json"))
        )

    def list_tags(self) -> List[str]:
        if not os.path.isdir(self.base_dir):
            return []
        return sorted(
            t for t in os.listdir(self.base_dir)
            if os.path.isfile(os.path.join(self.base_dir, t, "metadata.json"))
        )

    # ---------------------------------------------------------------- #
    #  保存                                                             #
    # ---------------------------------------------------------------- #
    def save(
        self,
        tag: str,
        all_h: List[np.ndarray],
        all_ds: List[Optional[List[DsLlm]]],
        sim_stats: SimulationStats,
        contexts: Optional[List[np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """保存冷启动仿真数据到磁盘。

        Args:
            tag:        缓存标签 (目录名)
            all_h:      LLM hidden states, 每个 shape (h_dim,)
            all_ds:     LLM 预测列表, 每项为 List[DsLlm] 或 None
            sim_stats:  预填充后的 SimulationStats
            contexts:   原始上下文特征 (用于验证)
            metadata:   额外元信息 (model_name, dataset, etc.)

        Returns:
            缓存目录路径
        """
        d = self.tag_dir(tag)
        os.makedirs(d, exist_ok=True)

        # 1) all_h → npy
        h_arr = np.array(all_h, dtype=np.float32)
        np.save(os.path.join(d, "all_h.npy"), h_arr)

        # 2) ds_llm → json
        ds_serialized = []
        for ds_list in all_ds:
            if ds_list is None:
                ds_serialized.append(None)
            else:
                ds_serialized.append([
                    {
                        "arm_id": entry.arm_id,
                        "predicted_mean": float(entry.predicted_mean),
                        "predicted_std": float(entry.predicted_std),
                        "reasoning": getattr(entry, "reasoning", ""),
                        "raw_text": getattr(entry, "raw_text", ""),
                    }
                    for entry in ds_list
                ])
        with open(os.path.join(d, "ds_llm.json"), "w") as f:
            json.dump(ds_serialized, f, ensure_ascii=False, indent=1)

        # 3) sim_stats → npz
        np.savez(
            os.path.join(d, "sim_stats.npz"),
            sum_llm=sim_stats.sum_llm,
            t_s=sim_stats.t_s,
            sum_online=sim_stats.sum_online,
            n_online=sim_stats.n_online,
            total_t=np.array([sim_stats.total_t]),
        )

        # 4) contexts → npy (可选)
        if contexts is not None:
            ctx_arr = np.array(contexts, dtype=np.float32)
            np.save(os.path.join(d, "contexts.npy"), ctx_arr)

        # 5) metadata → json
        meta = {
            "tag": tag,
            "cold_start_n": len(all_h),
            "h_dim": int(h_arr.shape[1]) if h_arr.ndim == 2 else 0,
            "num_actions": sim_stats.K,
            "created_at": datetime.now().isoformat(),
            "sim_stats_summary": sim_stats.summary(),
        }
        if metadata:
            meta.update(metadata)
        with open(os.path.join(d, "metadata.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return d

    # ---------------------------------------------------------------- #
    #  加载                                                             #
    # ---------------------------------------------------------------- #
    def load(self, tag: str) -> Dict[str, Any]:
        """加载冷启动缓存。

        Returns:
            {
                "all_h":     List[np.ndarray],  每个 shape (h_dim,)
                "all_ds":    List[Optional[List[DsLlm]]],
                "sim_stats_arrays": {sum_llm, t_s, sum_online, n_online, total_t},
                "contexts":  Optional[np.ndarray],   (N, context_dim) or None
                "metadata":  dict,
            }
        """
        d = self.tag_dir(tag)
        if not self.exists(tag):
            raise FileNotFoundError(f"冷启动缓存不存在: {d}")

        # 1) all_h
        h_arr = np.load(os.path.join(d, "all_h.npy"))
        all_h = [h_arr[i] for i in range(h_arr.shape[0])]

        # 2) ds_llm
        with open(os.path.join(d, "ds_llm.json"), "r") as f:
            ds_raw = json.load(f)
        all_ds: List[Optional[List[DsLlm]]] = []
        for ds_list_raw in ds_raw:
            if ds_list_raw is None:
                all_ds.append(None)
            else:
                all_ds.append([
                    DsLlm(
                        arm_id=e["arm_id"],
                        predicted_mean=e["predicted_mean"],
                        predicted_std=e.get("predicted_std", 1.0),
                        reasoning=e.get("reasoning", ""),
                        raw_text=e.get("raw_text", "")
                    )
                    for e in ds_list_raw
                ])

        # 3) sim_stats arrays
        ss_data = np.load(os.path.join(d, "sim_stats.npz"))
        sim_stats_arrays = {
            "sum_llm": ss_data["sum_llm"],
            "t_s": ss_data["t_s"],
            "sum_online": ss_data["sum_online"],
            "n_online": ss_data["n_online"],
            "total_t": int(ss_data["total_t"][0]),
        }

        # 4) contexts
        ctx_path = os.path.join(d, "contexts.npy")
        contexts = np.load(ctx_path) if os.path.isfile(ctx_path) else None

        # 5) metadata
        with open(os.path.join(d, "metadata.json"), "r") as f:
            metadata = json.load(f)

        return {
            "all_h": all_h,
            "all_ds": all_ds,
            "sim_stats_arrays": sim_stats_arrays,
            "contexts": contexts,
            "metadata": metadata,
        }

    def restore_sim_stats(
        self,
        sim_stats: SimulationStats,
        cache_data: Dict[str, Any],
    ):
        """从缓存数据恢复 SimulationStats 状态。

        Args:
            sim_stats:  目标 SimulationStats 对象
            cache_data: load() 返回的字典
        """
        arrays = cache_data["sim_stats_arrays"]
        sim_stats.sum_llm[:] = arrays["sum_llm"]
        sim_stats.t_s[:] = arrays["t_s"]
        sim_stats.sum_online[:] = arrays["sum_online"]
        sim_stats.n_online[:] = arrays["n_online"]
        sim_stats.total_t = arrays["total_t"]

    def build_sim_dataset(
        self,
        cache_data: Dict[str, Any],
    ):
        """从缓存数据重建 InMemoryOfflineDataset (仿真数据集)。

        Returns:
            InMemoryOfflineDataset 实例
        """
        from OurMethod.core.offline_dataset import InMemoryOfflineDataset

        all_h = cache_data["all_h"]
        all_ds = cache_data["all_ds"]
        h_dim = all_h[0].shape[0] if all_h else 0
        ds = InMemoryOfflineDataset(h_dim=h_dim)

        for i_ctx in range(len(all_ds)):
            ds_list = all_ds[i_ctx]
            h_t = all_h[i_ctx]
            if ds_list:
                for entry in ds_list:
                    ds.add(h_t, entry.arm_id, entry.predicted_mean)
        return ds

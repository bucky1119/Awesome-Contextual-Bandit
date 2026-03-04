"""Step 6 — Prompt-to-Vector Cache。

key   : hash(user/machine + arms_signature + feedback_bucket)
value : CacheEntry(h_t, z_t, timestamp, protect_flag)

淘汰策略：LRU（保护项不被驱逐，除非所有项受保护且超容量时才驱逐最旧保护项）。
监控：命中率、平均延迟、cache size。
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np


@dataclass
class CacheEntry:
    h_t: Optional[np.ndarray] = None
    z_t: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    protect: bool = False


class PromptVectorCache:
    """LRU Cache + protect_flag 保护策略。"""

    def __init__(self, capacity: int = 1024):
        if capacity <= 0:
            raise ValueError(f"capacity 必须 > 0, 得到 {capacity}")
        self.capacity = capacity
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._total_get_ns: int = 0
        self._eviction_count = 0

    # ---- key 生成 ---- #
    @staticmethod
    def make_key(
        user_id: Optional[str],
        machine_id: Optional[str],
        arms_signature: str,
        feedback_bucket: str = "",
    ) -> str:
        raw = f"{user_id or ''}|{machine_id or ''}|{arms_signature}|{feedback_bucket}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    # ---- 核心接口 ---- #
    def get(self, key: str) -> Optional[CacheEntry]:
        t0 = time.perf_counter_ns()
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            self._total_get_ns += time.perf_counter_ns() - t0
            return self._store[key]
        self._misses += 1
        self._total_get_ns += time.perf_counter_ns() - t0
        return None

    def put(
        self, key: str,
        h_t: Optional[np.ndarray] = None,
        z_t: Optional[np.ndarray] = None,
        protect: bool = False,
    ):
        if key in self._store:
            self._store.move_to_end(key)
            entry = self._store[key]
            entry.h_t = h_t
            entry.z_t = z_t
            entry.timestamp = time.time()
            entry.protect = protect
            return

        # 需要驱逐？
        while len(self._store) >= self.capacity:
            self._evict_one()

        self._store[key] = CacheEntry(
            h_t=h_t, z_t=z_t, timestamp=time.time(), protect=protect,
        )

    def _evict_one(self):
        """驱逐最旧的 非保护 项；若全部受保护则驱逐最旧保护项。"""
        # 尝试驱逐非保护项
        for k in list(self._store.keys()):
            if not self._store[k].protect:
                del self._store[k]
                self._eviction_count += 1
                return
        # 全部保护，驱逐最旧
        if self._store:
            self._store.popitem(last=False)
            self._eviction_count += 1

    def contains(self, key: str) -> bool:
        return key in self._store

    def set_protect(self, key: str, flag: bool):
        if key in self._store:
            self._store[key].protect = flag

    def invalidate(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self):
        self._store.clear()
        self._hits = self._misses = 0
        self._total_get_ns = 0
        self._eviction_count = 0

    # ---- 监控 ---- #
    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def avg_get_latency_us(self) -> float:
        total = self._hits + self._misses
        return (self._total_get_ns / total / 1000.0) if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "avg_get_latency_us": self.avg_get_latency_us,
            "evictions": self._eviction_count,
        }

    def __repr__(self) -> str:
        return f"PromptVectorCache(cap={self.capacity}, size={self.size}, hit={self.hit_rate:.1%})"

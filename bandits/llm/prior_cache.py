"""磁盘先验分数缓存模块

本模块提供基于磁盘的键值缓存，用于存储LLM生成的先验分数。

设计特点:
  - 使用pickle持久化，支持复杂Python对象
  - 线程安全，支持多线程并发访问
  - 原子写入，使用临时文件+rename避免损坏
  - 键基于内容哈希生成，确保相同输入得到相同键

使用场景:
  - LLM推理耗时，缓存避免重复计算
  - 多轮实验共享先验结果
  - 调试时查看历史推理结果
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import threading
from typing import Any, Dict, Iterable, Optional

import numpy as np


class DiskPriorCache:
  """磁盘先验分数缓存

  简单的磁盘键值缓存，用于存储先验分数。

  工作原理:
    - 使用SHA1哈希作为键 (基于样本ID、候选动作、提示风格等)
    - 值存储为numpy数组
    - 自动持久化到磁盘

  主要方法:
    - build_key(): 根据输入构建缓存键
    - get(): 获取缓存的先验分数
    - set(): 存储先验分数到缓存
  """

  def __init__(self, cache_dir: str, dataset_name: str, rebuild: bool = False):
    self.cache_dir = cache_dir
    self.dataset_name = dataset_name
    self.cache_file = os.path.join(
        cache_dir, "{}_prior_cache.pkl".format(self._sanitize(dataset_name)))
    self._lock = threading.Lock()
    self._data: Dict[str, Any] = {}

    os.makedirs(self.cache_dir, exist_ok=True)
    if rebuild and os.path.exists(self.cache_file):
      os.remove(self.cache_file)

    if os.path.exists(self.cache_file):
      try:
        with open(self.cache_file, "rb") as f:
          loaded = pickle.load(f)
        if isinstance(loaded, dict):
          self._data = loaded
      except Exception:
        self._data = {}

  @staticmethod
  def _sanitize(name: str) -> str:
    """清理文件名，移除非法字符

    参数:
      name: 原始名称

    返回:
      清理后的安全文件名
    """
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(name))

  @staticmethod
  def _hash_payload(payload: Dict[str, Any]) -> str:
    """将payload字典转换为SHA1哈希

    使用JSON序列化确保跨平台一致性:
      - sort_keys=True: 键排序保证相同内容产生相同哈希
      - ensure_ascii=False: 保留Unicode字符

    参数:
      payload: 要哈希的字典

    返回:
      40位SHA1十六进制字符串
    """
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

  def build_key(
      self,
      sample_id: Any,
      candidate_actions: Iterable[str],
      prompt_style: str,
      extra: Optional[Dict[str, Any]] = None,
  ) -> str:
    """构建缓存键

    键的组成:
      - dataset_name: 数据集名称
      - sample_id: 样本ID
      - candidate_actions: 候选动作列表 (完整文本)
      - prompt_style: 提示风格
      - extra: 额外信息 (如context的哈希值)

    参数:
      sample_id: 样本标识符
      candidate_actions: 候选动作文本列表
      prompt_style: 提示风格
      extra: 额外键值对 (可选)

    返回:
      SHA1哈希字符串作为缓存键
    """
    payload: Dict[str, Any] = {
      "dataset_name": self.dataset_name,
      "sample_id": str(sample_id),
      "candidate_actions": list(candidate_actions),
      "prompt_style": prompt_style,
    }
    if extra:
      payload.update(extra)
    return self._hash_payload(payload)

  def get(self, key: str) -> Optional[np.ndarray]:
    """从缓存获取先验分数

    参数:
      key: 缓存键 (由build_key生成)

    返回:
      先验分数数组，如果键不存在或解析失败返回None
    """
    value = self._data.get(key)
    if value is None:
      return None
    try:
      arr = np.asarray(value, dtype=np.float64)
      return arr
    except Exception:
      return None

  def set(self, key: str, scores: np.ndarray) -> None:
    """存储先验分数到缓存

    线程安全方法，使用锁保护:
      1. 将分数转换为列表存储
      2. 调用持久化方法写入磁盘

    参数:
      key: 缓存键
      scores: 先验分数numpy数组
    """
    with self._lock:
      self._data[key] = np.asarray(scores, dtype=np.float64).tolist()
      self._persist_locked()

  def _persist_locked(self) -> None:
    """原子写入持久化文件

    使用临时文件+rename确保原子性:
      1. 创建临时文件
      2. 写入pickle数据
      3. atomic rename覆盖旧文件

    这样即使写入过程中断也不会损坏旧缓存。
    """
    os.makedirs(self.cache_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="prior_cache_", suffix=".tmp", dir=self.cache_dir)
    try:
      with os.fdopen(fd, "wb") as f:
        pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)
      os.replace(tmp_path, self.cache_file)
    finally:
      if os.path.exists(tmp_path):
        os.remove(tmp_path)

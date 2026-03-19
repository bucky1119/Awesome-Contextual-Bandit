"""本地LLM先验提供者模块

本模块提供从本地LLM获取先验分数的功能，用于上下文 bandit 算法的先验输入。

主要组件:
  - UniformPriorProvider: 均匀分布先验 (所有动作概率相同)
  - PrecomputedPriorProvider: 从预计算文件加载先验
  - LocalLLMPriorProvider: 使用本地冻结LLM生成先验 (带磁盘缓存)

使用场景:
  在 bandit 算法中，LLM可以作为一个"先验知识来源"，
  帮助算法在冷启动阶段做出更好的决策。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import numpy as np

from bandits.llm.prior_cache import DiskPriorCache
from bandits.llm.prompt_registry import build_prior_prompt


class UniformPriorProvider:
  """均匀分布先验提供者

  这是一个简单的fallback提供者，返回均匀分布的先验概率。
  所有动作的先验分数相同 (1/k)，表示没有任何先验知识。

  适用场景:
    - 没有可用先验时的默认选项
    - 对照实验中的基准线
  """

  def __init__(self, dataset_name: str):
    self.dataset_name = dataset_name

  def get_prior_scores(
      self,
      sample_id: Any,
      context: np.ndarray,
      action_texts: Iterable[str],
      context_text: Optional[str] = None,
  ) -> np.ndarray:
    """获取均匀分布的先验分数

    参数:
      sample_id: 样本ID (此处未使用)
      context: 上下文向量 (此处未使用)
      action_texts: 候选动作的文本描述列表
      context_text: 上下文文本 (可选，用于文本数据集)

    返回:
      先验分数数组，形状 (k,)，所有值为 1/k
    """
    k = len(list(action_texts))
    if k <= 0:
      return np.zeros(0, dtype=np.float64)
    return np.ones(k, dtype=np.float64) / float(k)


class PrecomputedPriorProvider:
  """预计算先验提供者

  从预计算文件( JSON / NPZ / NPY格式 )加载先验分数。
  文件格式:
    - JSON: {"sample_id": [score_0, score_1, ...], ...}
    - NPZ: priors[n] = [scores for sample n]
    - NPY: 数组 shape = (n_samples, n_actions)

  特点:
    - 加载速度快，适合大规模实验
    - 允许使用其他方法预计算的先验
    - 如果sample_id不存在，返回均匀分布
  """

  def __init__(self, dataset_name: str, prior_path: str):
    self.dataset_name = dataset_name
    self.prior_path = prior_path
    self._table = self._load_table(prior_path)

  def _load_table(self, prior_path: str) -> Dict[str, Any]:
    if not prior_path or not os.path.exists(prior_path):
      raise ValueError("Precomputed prior file not found: {}".format(prior_path))

    lower = prior_path.lower()
    if lower.endswith(".json"):
      with open(prior_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
      if not isinstance(raw, dict):
        raise ValueError("JSON precomputed priors must be a dict keyed by sample_id")
      return raw

    if lower.endswith(".npy"):
      arr = np.load(prior_path, allow_pickle=True)
      if isinstance(arr, np.ndarray) and arr.dtype != object:
        return {str(i): arr[i].tolist() for i in range(arr.shape[0])}
      obj = arr.item()
      if isinstance(obj, dict):
        return obj
      raise ValueError("Unsupported .npy precomputed priors format")

    if lower.endswith(".npz"):
      d = np.load(prior_path, allow_pickle=True)
      if "priors" in d:
        priors = d["priors"]
        return {str(i): priors[i].tolist() for i in range(priors.shape[0])}
      return {k: d[k].tolist() for k in d.files}

    raise ValueError("Unsupported precomputed prior file type: {}".format(prior_path))

  def get_prior_scores(
      self,
      sample_id: Any,
      context: np.ndarray,
      action_texts: Iterable[str],
      context_text: Optional[str] = None,
  ) -> np.ndarray:
    key = str(sample_id)
    k = len(list(action_texts))
    value = self._table.get(key)
    if value is None:
      return np.ones(k, dtype=np.float64) / max(float(k), 1.0)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != k:
      return np.ones(k, dtype=np.float64) / max(float(k), 1.0)
    return np.clip(arr, 0.0, 1.0)


class LocalLLMPriorProvider:
  """本地冻结LLM先验提供者 (带磁盘缓存)

  使用本地部署的冻结语言模型生成先验分数。
  核心工作流程:
    1. 为每个样本构建分类任务提示
    2. 使用LLM生成候选动作的排序
    3. 将排序转换为分数 (排名第1得1分，排名最后得0分)
    4. 缓存结果到磁盘避免重复计算

  输入:
    - context: 归一化的数值特征向量
    - action_texts: 候选类别的文本描述
    - context_text: (可选) 原始文本上下文

  输出:
    - 先验分数数组，范围 [0, 1]，总和不一定为1

  特点:
    - 支持多种提示风格 (text_classification, numeric_feature等)
    - 磁盘缓存机制避免重复LLM调用
    - 推理时使用greedy decoding (temperature=0)
  """

  def __init__(
      self,
      dataset_name: str,
      model_path: str,
      tokenizer_path: Optional[str] = None,
      device: str = "auto",
      dtype: str = "auto",
      cache_dir: str = "results/prior_cache",
      rebuild_cache: bool = False,
      prompt_style: str = "auto",
      max_new_tokens: int = 128,
      dump_prompts: int = 0,
      debug_dir: str = "results/prior_debug",
      run_tag: Optional[str] = None,
  ):
    """初始化本地LLM先验提供者

    参数:
      dataset_name: 数据集名称，用于缓存区分
      model_path: 本地模型路径或HuggingFace模型ID
      tokenizer_path: (可选) tokenizer路径，默认同model_path
      device: 运行设备，"auto"/"cuda"/"cpu"，默认自动选择
      dtype: 模型精度，"auto"/"float32"/"float16"/"bfloat16"，默认自动选择
      cache_dir: 缓存目录路径
      rebuild_cache: 是否重建缓存 (删除旧缓存)
      prompt_style: 提示风格，"auto"/"text_classification"/"numeric_feature"等
      max_new_tokens: LLM最大生成token数
      dump_prompts: 前N个样本的提示词保存到调试目录 (用于分析)
      debug_dir: 调试信息保存目录
      run_tag: 运行标签，用于调试文件名区分
    """
    if not model_path:
      raise ValueError("Local LLM model path/id is empty")

    self.dataset_name = dataset_name
    self.model_path = model_path
    self.tokenizer_path = tokenizer_path or model_path
    self.device = device
    self.dtype = dtype
    self.prompt_style = prompt_style
    self.max_new_tokens = max_new_tokens
    self.cache = DiskPriorCache(cache_dir=cache_dir, dataset_name=dataset_name, rebuild=rebuild_cache)
    self.dump_prompts = max(0, int(dump_prompts))
    self._dump_count = 0
    self._dump_banner_printed = False
    self._run_tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    self.debug_dir = os.path.join(debug_dir, dataset_name)
    os.makedirs(self.debug_dir, exist_ok=True)

    self._load_model_and_tokenizer()

  def _load_model_and_tokenizer(self) -> None:
    try:
      import torch
      from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
      raise ImportError(
          "transformers/torch is required for LocalLLMPriorProvider. "
          "Please install dependencies in requirements.txt"
      ) from exc

    self._torch = torch

    tokenizer_path = self.tokenizer_path

    self.tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=True,
    )

    target_dtype = self._resolve_torch_dtype(torch)
    device_map = self._resolve_device_map(torch)

    self.model = AutoModelForCausalLM.from_pretrained(
        self.model_path,
        local_files_only=True,
        torch_dtype=target_dtype,
        device_map=device_map,
    )

    self.model.eval()
    for p in self.model.parameters():
      p.requires_grad = False

  def _resolve_torch_dtype(self, torch):
    value = (self.dtype or "auto").lower()
    if value in {"float32", "fp32"}:
      return torch.float32
    if value in {"float16", "fp16"}:
      return torch.float16
    if value in {"bfloat16", "bf16"}:
      return torch.bfloat16

    # auto
    if torch.cuda.is_available():
      if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
      return torch.float16
    return torch.float32

  def _resolve_device_map(self, torch):
    value = (self.device or "auto").lower()
    if value == "cpu":
      return "cpu"
    if value == "cuda":
      if torch.cuda.is_available():
        return "auto"
      return "cpu"
    return "auto" if torch.cuda.is_available() else "cpu"

  def get_prior_scores(
      self,
      sample_id: Any,
      context: np.ndarray,
      action_texts: Iterable[str],
      context_text: Optional[str] = None,
  ) -> np.ndarray:
    action_list = list(action_texts)
    k = len(action_list)
    if k == 0:
      return np.zeros(0, dtype=np.float64)
    if k == 1:
      return np.ones(1, dtype=np.float64)

    key = self.cache.build_key(
        sample_id=sample_id,
        candidate_actions=action_list,
        prompt_style=self.prompt_style,
        extra={
            "context_sha1": self._context_sha1(context),
            "has_text": context_text is not None,
        },
    )
    cached = self.cache.get(key)
    if cached is not None and cached.shape[0] == k:
      if self._should_dump_prompt():
        prompt = build_prior_prompt(
            dataset_name=self.dataset_name,
            context=context,
            action_texts=action_list,
            sample_id=int(sample_id),
            prompt_style=self.prompt_style,
            context_text=context_text,
        )
        self._dump_prompt_debug(
            sample_id=sample_id,
            prompt=prompt,
            action_texts=action_list,
            llm_output=None,
            ranking=None,
            scores=np.clip(cached, 0.0, 1.0),
            cache_hit=True,
            error_text=None,
        )
      return np.clip(cached, 0.0, 1.0)

    prompt = build_prior_prompt(
        dataset_name=self.dataset_name,
        context=context,
        action_texts=action_list,
        sample_id=int(sample_id),
        prompt_style=self.prompt_style,
        context_text=context_text,
    )

    try:
      ranking, llm_output = self._infer_ranking(prompt, action_list)
      scores = self._ranking_to_scores(ranking, k)
      error_text = None
    except Exception:
      ranking = None
      llm_output = None
      error_text = "llm_inference_or_parsing_failed"
      scores = np.ones(k, dtype=np.float64) / float(k)

    self.cache.set(key, scores)
    if self._should_dump_prompt():
      self._dump_prompt_debug(
          sample_id=sample_id,
          prompt=prompt,
          action_texts=action_list,
          llm_output=llm_output,
          ranking=ranking,
          scores=np.clip(scores, 0.0, 1.0),
          cache_hit=False,
          error_text=error_text,
      )
    return np.clip(scores, 0.0, 1.0)

  def _context_sha1(self, context: np.ndarray) -> str:
    arr = np.asarray(context, dtype=np.float32).reshape(-1)
    return hashlib.sha1(arr.tobytes()).hexdigest()

  def _infer_ranking(self, prompt: str, action_texts: list[str]) -> tuple[list[int], str]:
    torch = self._torch
    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
    model_device = self.model.device
    for key, value in inputs.items():
      inputs[key] = value.to(model_device)

    with torch.no_grad():
      output_ids = self.model.generate(
          **inputs,
          max_new_tokens=self.max_new_tokens,
          do_sample=False,
          temperature=0.0,
          pad_token_id=self.tokenizer.eos_token_id,
      )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    ranking = self._parse_ranking(text, action_texts)
    if len(ranking) != len(action_texts):
      raise ValueError("Parsed ranking is incomplete")
    return ranking, text

  def _should_dump_prompt(self) -> bool:
    return self._dump_count < self.dump_prompts

  def _dump_prompt_debug(
      self,
      sample_id: Any,
      prompt: str,
      action_texts: list[str],
      llm_output: Optional[str],
      ranking: Optional[list[int]],
      scores: np.ndarray,
      cache_hit: bool,
      error_text: Optional[str],
  ) -> None:
    dump_idx = self._dump_count
    dump_path = os.path.join(
        self.debug_dir,
        "{}_prompt_{:02d}_sample{}.txt".format(self._run_tag, dump_idx, sample_id),
    )

    with open(dump_path, "w", encoding="utf-8") as f:
      f.write("=== PRIOR DEBUG ===\n")
      f.write("dataset: {}\n".format(self.dataset_name))
      f.write("sample_id: {}\n".format(sample_id))
      f.write("cache_hit: {}\n".format(cache_hit))
      if error_text is not None:
        f.write("error: {}\n".format(error_text))
      f.write("\n=== ACTIONS ===\n")
      for i, action in enumerate(action_texts):
        f.write("{}: {}\n".format(i, action))

      f.write("\n=== PROMPT ===\n")
      f.write(prompt)

      f.write("\n\n=== LLM OUTPUT ===\n")
      if llm_output is None:
        f.write("None\n")
      else:
        f.write(llm_output)
        f.write("\n")

      f.write("\n=== PARSED RANKING ===\n")
      if ranking is None:
        f.write("None\n")
      else:
        f.write(json.dumps(ranking, ensure_ascii=False))
        f.write("\n")

      f.write("\n=== PRIOR SCORES ===\n")
      f.write(json.dumps([float(x) for x in scores.tolist()], ensure_ascii=False))
      f.write("\n")

    self._dump_count += 1
    if self._dump_count >= self.dump_prompts and not self._dump_banner_printed:
      print(
          "[DEBUG] Dumped {} prior prompts to {}".format(
              self.dump_prompts,
              self.debug_dir,
          )
      )
      self._dump_banner_printed = True

  def _parse_ranking(self, text: str, action_texts: list[str]) -> list[int]:
    k = len(action_texts)

    # 1) JSON block: {"ranking": [..]}
    json_candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for jc in json_candidates:
      try:
        obj = json.loads(jc)
      except Exception:
        continue
      if not isinstance(obj, dict) or "ranking" not in obj:
        continue
      parsed = self._normalize_rank_items(obj["ranking"], action_texts)
      if len(parsed) == k:
        return parsed

    # 2) Plain list like [0,2,1]
    list_candidates = re.findall(r"\[[^\[\]]+\]", text)
    for lc in list_candidates:
      try:
        obj = json.loads(lc)
      except Exception:
        continue
      parsed = self._normalize_rank_items(obj, action_texts)
      if len(parsed) == k:
        return parsed

    # 3) Fallback parse from tokens/lines
    tokens = re.split(r"[\n,;>\-]+", text)
    parsed = self._normalize_rank_items(tokens, action_texts)
    if len(parsed) == k:
      return parsed

    raise ValueError("Unable to parse ranking from LLM output")

  def _normalize_rank_items(self, raw_items: Any, action_texts: list[str]) -> list[int]:
    k = len(action_texts)
    if not isinstance(raw_items, (list, tuple)):
      return []

    used = set()
    out = []

    lower_text_to_idx = {action_texts[i].lower(): i for i in range(k)}
    for item in raw_items:
      idx = self._item_to_index(item, lower_text_to_idx, k)
      if idx is None or idx in used:
        continue
      used.add(idx)
      out.append(idx)

    return out

  def _item_to_index(self, item: Any, text_to_idx: Dict[str, int], k: int) -> Optional[int]:
    if isinstance(item, (int, np.integer)):
      val = int(item)
      if 0 <= val < k:
        return val
      if 1 <= val <= k:
        return val - 1
      return None

    s = str(item).strip().lower()
    if s in text_to_idx:
      return text_to_idx[s]

    m = re.search(r"\d+", s)
    if m:
      val = int(m.group())
      if 0 <= val < k:
        return val
      if 1 <= val <= k:
        return val - 1

    # try fuzzy match if label appears as substring
    for label, idx in text_to_idx.items():
      if label in s:
        return idx
    return None

  def _ranking_to_scores(self, ranking: list[int], k: int) -> np.ndarray:
    scores = np.zeros(k, dtype=np.float64)
    if k == 1:
      scores[0] = 1.0
      return scores

    for rank_pos, action_idx in enumerate(ranking):
      scores[action_idx] = 1.0 - (float(rank_pos) / float(k - 1))

    if np.all(scores == 0.0):
      return np.ones(k, dtype=np.float64) / float(k)
    return scores

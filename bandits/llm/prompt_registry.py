"""提示词注册表模块

本模块提供数据集相关的提示词模板构建功能。

主要功能:
  - get_action_texts(): 获取数据集的标准动作名称
  - get_dataset_prompt_template_name(): 根据数据集类型选择提示模板
  - build_prior_prompt(): 构建完整的LLM分类排序提示

支持的提示风格:
  - text_classification: 文本分类任务 (ag_news, newsgroups)
  - numeric_feature: 数值特征分类任务 (statlog, magic, mnist等)
  - generic_ranking: 通用排序提示

数据流:
  原始数据 -> 归一化特征 -> 构建提示 -> LLM推理 -> 排序结果 -> 先验分数
"""

from __future__ import annotations

from typing import List

import numpy as np


# 文本分类数据集 (使用原始文本作为上下文)
_TEXT_DATASETS = {"ag_news", "newsgroups"}

# 数值特征数据集 (使用归一化数值向量作为上下文)
_NUMERIC_DATASETS = {
    "statlog",
    "statlog_shuttle",
    "magic",
    "mnist",
    "adult",
    "census",
    "cencus",
    "covertype",
}

# 数据集的标准动作/类别名称映射
_DATASET_ACTIONS = {
    "ag_news": ["World", "Sports", "Business", "Sci/Tech"],  # 4类新闻
    "newsgroups": [
        "comp.graphics",
        "rec.sport.baseball",
        "sci.med",
        "sci.space",
        "talk.politics.guns",
        "soc.religion.christian",
    ],  # 6类新闻组
    "magic": ["gamma", "hadron"],  # 二分类: 伽马射线 vs 强子
    "adult": ["income_low", "income_high"],  # 收入是否超过50K
}


def get_action_texts(dataset_name: str, num_actions: int, prompt_style: str = "auto") -> List[str]:
  """获取数据集的标准动作/类别名称

  根据数据集名称返回人类可读的类别名称，用于构建LLM提示。

  参数:
    dataset_name: 数据集名称
    num_actions: 动作/类别数量
    prompt_style: 提示风格 (当前未使用，预留)

  返回:
    动作名称列表，如 ["World", "Sports", "Business", "Sci/Tech"]

  回退逻辑:
    - mnist: digit_0, digit_1, ...
    - statlog: shuttle_state_0, shuttle_state_1, ...
    - covertype: cover_type_1, cover_type_2, ... (1-based)
    - census: census_class_0, census_class_1, ...
    - 其他: class_0, class_1, ...
  """
  dataset_name = (dataset_name or "").lower()
  preset = _DATASET_ACTIONS.get(dataset_name)
  if preset and len(preset) == num_actions:
    return preset

  if dataset_name in {"mnist"}:
    return ["digit_{}".format(i) for i in range(num_actions)]
  if dataset_name in {"statlog", "statlog_shuttle"}:
    return ["shuttle_state_{}".format(i) for i in range(num_actions)]
  if dataset_name in {"covertype"}:
    return ["cover_type_{}".format(i + 1) for i in range(num_actions)]
  if dataset_name in {"census", "cencus"}:
    return ["census_class_{}".format(i) for i in range(num_actions)]

  return ["class_{}".format(i) for i in range(num_actions)]


def get_dataset_prompt_template_name(dataset_name: str) -> str:
  """根据数据集类型获取对应的提示模板名称

  参数:
    dataset_name: 数据集名称

  返回:
    提示模板名称:
      - "text_classification_ranking": 文本分类任务
      - "numeric_feature_ranking": 数值特征分类任务
      - "generic_ranking": 通用排序任务
  """
  dataset_name = (dataset_name or "").lower()
  if dataset_name in _TEXT_DATASETS:
    return "text_classification_ranking"
  if dataset_name in _NUMERIC_DATASETS:
    return "numeric_feature_ranking"
  return "generic_ranking"


def build_prior_prompt(
    dataset_name: str,
    context: np.ndarray,
    action_texts: List[str],
    sample_id: int,
    prompt_style: str = "auto",
    context_text: str | None = None,
) -> str:
  """构建用于LLM先验推理的分类排序提示

  根据数据集类型构建完整的提示，包括:
    1. 任务描述 (你是文本/数值分类助手)
    2. 数据集名称
    3. 候选类别列表 (带索引)
    4. 上下文信息 (文本或数值特征)
    5. 输出格式要求 (JSON格式的排序结果)

  参数:
    dataset_name: 数据集名称
    context: 归一化的上下文/特征向量
    action_texts: 候选动作/类别名称列表
    sample_id: 样本ID (用于调试)
    prompt_style: 提示风格 (当前未使用，预留)
    context_text: (可选) 原始文本上下文，用于文本数据集

  返回:
    完整的提示字符串，LLM输入

  输出格式示例:
    {
      "ranking": [0, 2, 1, 3]  // 第0类最可能，第3类最不可能
    }
  """
  dataset_name = (dataset_name or "").lower()
  k = len(action_texts)

  action_lines = "\n".join(
      "{}: {}".format(i, action_texts[i]) for i in range(k)
  )

  if context_text and dataset_name in _TEXT_DATASETS:
    context_block = (
        "Sample id: {}\n"
        "Input text:\n"
        "{}\n"
    ).format(sample_id, str(context_text).strip())
  else:
    context_block = _numeric_context_block(context)

  ranking_rule = (
      "Return a complete ranking over ALL candidate labels as JSON only: "
      "{\"ranking\": [label_index_0, label_index_1, ...]}. "
      "Use 0-based label indices and include each index exactly once."
  )
  if k == 1:
    ranking_rule = "Only one label is available. Return {\"ranking\": [0]} as JSON only."
  elif k == 2:
    ranking_rule += " This is binary classification, still output both labels in ranked order."

  if dataset_name in {"ag_news", "newsgroups"}:
    task = "You are a text classification assistant."
  elif dataset_name == "mnist":
    task = "You are a digit classification assistant using vectorized image features."
  elif dataset_name in {"magic", "statlog", "statlog_shuttle", "covertype"}:
    task = "You are a tabular classification assistant for scientific sensor-style features."
  else:
    task = "You are a tabular classification assistant."

  prompt = (
      "{}\n"
      "Dataset: {}\n"
      "Candidate labels:\n"
      "{}\n\n"
      "{}\n"
      "{}\n"
  ).format(task, dataset_name or "unknown", action_lines, context_block, ranking_rule)

  return prompt


def _numeric_context_block(context: np.ndarray, max_features: int = 40) -> str:
  """将数值特征向量转换为LLM可读的文本描述

  将归一化的数值特征转换为格式化的字符串列表:
    f0=0.1234, f1=0.5678, f2=0.9012, ...

  参数:
    context: 归一化的特征向量
    max_features: 最大显示的特征数量 (避免提示过长)

  返回:
    特征描述字符串

  注意:
    - 只显示前max_features个特征 (默认40)
    - 如果特征更多，会添加提示说明
    - 特征值保留4位小数
  """
  flat = np.asarray(context, dtype=np.float64).reshape(-1)
  n = min(max_features, flat.shape[0])
  feats = ["f{}={:.4f}".format(i, float(flat[i])) for i in range(n)]
  tail = ""
  if flat.shape[0] > n:
    tail = "\n(Only first {} of {} normalized features are shown.)".format(n, flat.shape[0])

  return (
      "This sample has the following normalized numeric features:\n"
      "{}{}"
  ).format(", ".join(feats), tail)

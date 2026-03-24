#!/usr/bin/env python3
"""
通用分步实验脚本：支持多数据集、多 baseline、OurMethod 与灵活绘图。

示例:
  # 1) 跑 baseline（可多数据集）
  python run_dataset_step.py baselines --datasets newsgroups statlog --n_rounds 2000 --seed 42

  # 2) 跑 OurMethod（结果名包含模型）
  python run_dataset_step.py ourmethod --datasets newsgroups --model qwen2_5_7b --n_rounds 2000 --seed 42

  # 3) 绘图
  # 3.1 画 7 个 baseline
  python run_dataset_step.py plot --datasets newsgroups --algorithms all_baselines --n_rounds 2000 --seed 42

  # 3.2 单算法绘图
  python run_dataset_step.py plot --datasets newsgroups --algorithms neural_linear --n_rounds 2000 --seed 42

  # 3.3 指定 baseline + OurMethod
  python run_dataset_step.py plot --datasets newsgroups --algorithms linucb ourmethod:smollm2 --n_rounds 2000 --seed 42
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import random as _random
import re
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))
from OurMethod.extract_ucb import extract_ucb_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _set_all_seeds(seed: int) -> None:
    """统一固定 random / numpy / torch 三类随机源，保证实验可复现。"""
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bandits.core.contextual_bandit import ContextualBandit
from bandits.data.data_sampler import (
    sample_newsgroups_data,
    sample_statlog_data,
    sample_statlog_shuttle_data,
    sample_ag_news_data,
    sample_magic_data,
    sample_mnist_data,
    sample_adult_data,
    sample_census_data,
    sample_covertype_data,
)

from bandits.algorithms.neural_bandit_model import NeuralBanditModel
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.algorithms.ucb1_sampling import UCB1Sampling
from bandits.algorithms.linucb_sampling import LinUCBSampling
from bandits.algorithms.epsilon_greedy_sampling import EpsilonGreedySampling
from bandits.algorithms.neural_ucb_sampling import NeuralUCBSampling
from bandits.algorithms.neural_linucb_sampling import NeuralLinUCBSampling
from bandits.algorithms.Inl_ucb_sampling import InlUCBSampling
from bandits.algorithms.ATResidualLinUCB import AdaptiveTrustResidualLinUCB
from bandits.llm.local_prior_provider import (
    LocalLLMPriorProvider,
    PrecomputedPriorProvider,
    UniformPriorProvider,
)
from bandits.llm.prompt_registry import (
    get_action_texts,
    get_dataset_prompt_template_name,
)
from experiment_logger import save_single_run_log, save_dataset_summary_log


RESULTS_DIR = os.path.join(ROOT, "results")


# ====================================================================== #
#  Registry                                                               #
# ====================================================================== #

DatasetLoader = Callable[[int, int, int], Tuple[ContextualBandit, np.ndarray, np.ndarray, int, int, int]]
BaselineFactory = Callable[[int, int], Any]

DATASET_REGISTRY: Dict[str, DatasetLoader] = {}
BASELINE_REGISTRY: Dict[str, BaselineFactory] = {}
BASELINE_RUNTIME_CONFIG: Dict[str, Any] = {}


def register_dataset(name: str, loader: DatasetLoader):
    DATASET_REGISTRY[name] = loader


def register_baseline(name: str, factory: BaselineFactory):
    BASELINE_REGISTRY[name] = factory


# ====================================================================== #
#  Dataset loaders                                                        #
# ====================================================================== #

def load_newsgroups(n_rounds: int, cold_start_n: int, seed: int):
    data_path = os.path.join(ROOT, "datasets", "newsgroups.npz")
    if not os.path.exists(data_path):
        print(f"[ERROR] {data_path} not found. Run prepare_newsgroups.py first.")
        sys.exit(1)

    _set_all_seeds(seed)
    total = n_rounds + cold_start_n
    dataset, (opt_r_all, opt_a_all), texts = sample_newsgroups_data(
        data_path,
        total,
        shuffle_rows=True,
        return_texts=True,
    )

    actual_total = len(opt_r_all)
    if total > actual_total:
        n_rounds = actual_total - cold_start_n
        print(f"[WARN] newsgroups adjusted n_rounds={n_rounds}")

    num_actions = 6
    context_dim = dataset.shape[1] - num_actions
    opt_rewards = opt_r_all[cold_start_n: cold_start_n + n_rounds]
    opt_actions = opt_a_all[cold_start_n: cold_start_n + n_rounds]

    cmab = ContextualBandit(num_actions, context_dim)
    cmab.feed_data(dataset)
    if texts is not None:
        cmab.texts = texts
    return cmab, opt_rewards, opt_actions, num_actions, context_dim, n_rounds


def load_statlog(n_rounds: int, cold_start_n: int, seed: int):
    data_path = os.path.join(ROOT, "datasets", "statlog.trn")
    if not os.path.exists(data_path):
        print(f"[ERROR] {data_path} not found.")
        sys.exit(1)

    _set_all_seeds(seed)
    total = n_rounds + cold_start_n
    dataset, (opt_r_all, opt_a_all) = sample_statlog_data(data_path, total, shuffle_rows=True)

    actual_total = len(opt_r_all)
    if total > actual_total:
        n_rounds = actual_total - cold_start_n
        print(f"[WARN] statlog adjusted n_rounds={n_rounds}")

    num_actions = 7
    context_dim = dataset.shape[1] - num_actions
    opt_rewards = opt_r_all[cold_start_n: cold_start_n + n_rounds]
    opt_actions = opt_a_all[cold_start_n: cold_start_n + n_rounds]

    cmab = ContextualBandit(num_actions, context_dim)
    cmab.feed_data(dataset)
    return cmab, opt_rewards, opt_actions, num_actions, context_dim, n_rounds


def load_ag_news(n_rounds: int, cold_start_n: int, seed: int):
    data_path = os.path.join(ROOT, "datasets", "ag_news_tfidf_svd100.npz")
    if not os.path.exists(data_path):
        print(f"[ERROR] {data_path} not found. Run prepare_ag_news.py first.")
        sys.exit(1)

    _set_all_seeds(seed)
    total = n_rounds + cold_start_n
    dataset, (opt_r_all, opt_a_all), texts = sample_ag_news_data(data_path, total, shuffle_rows=True, return_texts=True)

    actual_total = len(opt_r_all)
    if total > actual_total:
        n_rounds = actual_total - cold_start_n
        print(f"[WARN] ag_news adjusted n_rounds={n_rounds}")

    num_actions = 4
    context_dim = dataset.shape[1] - num_actions
    opt_rewards = opt_r_all[cold_start_n: cold_start_n + n_rounds]
    opt_actions = opt_a_all[cold_start_n: cold_start_n + n_rounds]

    cmab = ContextualBandit(num_actions, context_dim)
    cmab.feed_data(dataset)
    # Inject text sequence for OurMethod
    cmab.texts = texts

    return cmab, opt_rewards, opt_actions, num_actions, context_dim, n_rounds


def _load_generic_uci(
    sampler: Callable[[int, bool], Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]],
    n_rounds: int,
    cold_start_n: int,
    seed: int,
    num_actions: int | None = None,
):
    _set_all_seeds(seed)
    total = n_rounds + cold_start_n
    dataset, (opt_r_all, opt_a_all) = sampler(total, True)

    actual_total = len(opt_r_all)
    if total > actual_total:
        n_rounds = actual_total - cold_start_n
        print(f"[WARN] adjusted n_rounds={n_rounds}")

    if num_actions is None:
        num_actions = int(len(np.unique(opt_a_all)))
    context_dim = int(dataset.shape[1] - num_actions)

    opt_rewards = opt_r_all[cold_start_n: cold_start_n + n_rounds]
    opt_actions = opt_a_all[cold_start_n: cold_start_n + n_rounds]

    cmab = ContextualBandit(num_actions, context_dim)
    cmab.feed_data(dataset)
    return cmab, opt_rewards, opt_actions, num_actions, context_dim, n_rounds


def load_magic(n_rounds: int, cold_start_n: int, seed: int):
    return _load_generic_uci(sample_magic_data, n_rounds, cold_start_n, seed, num_actions=2)


def load_mnist(n_rounds: int, cold_start_n: int, seed: int):
    return _load_generic_uci(sample_mnist_data, n_rounds, cold_start_n, seed, num_actions=10)


def load_adult(n_rounds: int, cold_start_n: int, seed: int):
    return _load_generic_uci(sample_adult_data, n_rounds, cold_start_n, seed, num_actions=2)


def load_census(n_rounds: int, cold_start_n: int, seed: int):
    return _load_generic_uci(sample_census_data, n_rounds, cold_start_n, seed)


def load_covertype(n_rounds: int, cold_start_n: int, seed: int):
    return _load_generic_uci(sample_covertype_data, n_rounds, cold_start_n, seed, num_actions=7)


def load_statlog_shuttle(n_rounds: int, cold_start_n: int, seed: int):
    return _load_generic_uci(sample_statlog_shuttle_data, n_rounds, cold_start_n, seed, num_actions=7)

register_dataset("newsgroups", load_newsgroups)
register_dataset("statlog", load_statlog)
register_dataset("ag_news", load_ag_news)
register_dataset("magic", load_magic)
register_dataset("mnist", load_mnist)
register_dataset("adult", load_adult)
register_dataset("census", load_census)
register_dataset("cencus", load_census)
register_dataset("covertype", load_covertype)
register_dataset("statlog_shuttle", load_statlog_shuttle)


# ====================================================================== #
#  Baseline registry                                                      #
# ====================================================================== #

def _b_neural_bandit(num_actions: int, context_dim: int):
    h = {
        "context_dim": context_dim,
        "num_actions": num_actions,
        "layer_sizes": [100, 100],
        "activation": "relu",
        "initial_lr": 0.001,
        "batch_size": 512,
        "init_scale": 0.3,
        "use_dropout": False,
        "dropout_rate": 0.1,
        "layer_norm": False,
        "verbose": False,
    }
    return NeuralBanditModel(h, name="neural_bandit")


def _b_neural_linear(num_actions: int, context_dim: int):
    h = {
        "context_dim": context_dim,
        "num_actions": num_actions,
        "layer_sizes": [100, 100],
        "activation": "relu",
        "initial_lr": 0.001,
        "batch_size": 512,
        "init_scale": 0.3,
        "use_dropout": False,
        "dropout_rate": 0.1,
        "layer_norm": False,
        "verbose": False,
        "lambda_prior": 0.25,
        "a0": 6,
        "b0": 6,
        "training_freq": 50,
        "training_freq_network": 50,
        "training_epochs": 50,
        "initial_pulls": 2,
    }
    return NeuralLinearPosteriorSampling(h, name="neural_linear")


def _b_ucb1(num_actions: int, context_dim: int):
    h = argparse.Namespace(num_actions=num_actions, alpha=2.0)
    return UCB1Sampling("ucb1", h)


def _b_linucb(num_actions: int, context_dim: int):
    h = argparse.Namespace(num_actions=num_actions, context_dim=context_dim, alpha=1.0)
    return LinUCBSampling("linucb", h)


def _b_epsilon_greedy(num_actions: int, context_dim: int):
    h = argparse.Namespace(num_actions=num_actions, epsilon=0.1, epsilon_decay=0.0)
    return EpsilonGreedySampling("epsilon_greedy", h)


# def _b_neural_ucb(num_actions: int, context_dim: int):
#     h = {
#         "context_dim": context_dim,
#         "num_actions": num_actions,

#         # ===== 网络结构 =====
#         "layer_sizes": [100],        # m = 100（论文一致）
#         "activation": "relu",

#         # ===== 训练参数（论文核心）=====
#         "initial_lr": 0.01,          # 需要 grid search
#         "batch_size": 500,           # 论文固定
#         "training_epochs": 1000,     # J = 1000（关键！！）

#         # ===== 训练调度（非常关键）=====
#         "training_freq": 100,        # 每100轮训练一次（论文）
#         "training_starts_at": 2000,  # 从2000轮后开始训练（论文）

#         # ===== 正则 & UCB =====
#         "lambda_prior": 1e-3,        # ⚠️ 强烈建议不要用1.0
#         "alpha": 1.0,                # exploration 系数（需要调）

#         # ===== 其他 =====
#         "initial_pulls": 2,
#         "verbose": False,
#     }
#     return NeuralUCBSampling(h, name="neural_ucb")

def _b_neural_ucb(num_actions: int, context_dim: int):
    import os

    h = {
        "context_dim": context_dim,
        "num_actions": num_actions,

        "layer_sizes": [100],
        "activation": "relu",

        "initial_lr": float(os.getenv("NUCB_LR", "0.001")),
        "batch_size": 500,
        "training_epochs": int(os.getenv("NUCB_EPOCHS", "1000")),

        "training_freq": 100,
        "training_starts_at": 500, # 论文是2000

        "lambda_prior": float(os.getenv("NUCB_LAMBDA", "1e-2")),
        "alpha": float(os.getenv("NUCB_ALPHA", "0.1")),
        "initial_pulls": 2,
        "verbose": False,
        # grid search ：lam in 1e-2 1e-3;lr in 0.01 0.001; alpha in 0.1 1.0 5
        # 目前是lr=0.01,lambda_prior=1e-2, alpha=0.1效果最好，为2072的cumulative regret，后续可以继续调参
        # 目前是lr=0.001,lambda_prior=1e-2, alpha=0.1效果最好，为1887的cumulative regret，后续可以继续调参
    }
    return NeuralUCBSampling(h, name="neural_ucb")


def _b_neural_linucb(num_actions: int, context_dim: int):
    h = {
        "context_dim": context_dim,
        "num_actions": num_actions,
        "layer_sizes": [100],
        "activation": "relu",
        "initial_lr": 0.001,
        "batch_size": 512,
        "init_scale": 0.3,
        "use_dropout": False,
        "dropout_rate": 0.1,
        "layer_norm": False,
        "verbose": False,
        "lambda_prior": 1.0,
        "alpha": 1,
        # "training_freq": 50,
        "training_freq": 10,
        "training_epochs": 100,
    }
    return NeuralLinUCBSampling(h, name="neural_linucb")


def _b_inlucb(num_actions: int, context_dim: int):
    """InlUCB: Interconnected Neural-Linear UCB (Chen et al., 2022).

    Alternates between:
      - Online phase: fix f, run shared-linear UCB on latent features.
      - Offline phase: fix theta, train f by MSE loss.
    """
    h = {
        "context_dim": context_dim,
        "num_actions": num_actions,
        "layer_sizes": [100, 100],
        "activation": "relu",
        "initial_lr": 0.001,
        "batch_size": 256,
        "weight_decay": 1e-4,
        "use_dropout": False,
        "dropout_rate": 0.0,
        "layer_norm": False,
        "verbose": False,
        "alpha": 1.0,
        "lambda_prior": 1.0,
        "online_horizon": 100,   # T steps per iteration before offline update
        "offline_epochs": 50,    # epochs for offline representation learning
        "initial_pulls": 2,
    }
    return InlUCBSampling(h, name="inlucb")


def _b_at_residual_linucb(num_actions: int, context_dim: int):
    h = argparse.Namespace(
        num_actions=num_actions,
        context_dim=context_dim,
        alpha=1.0,
        lambda_prior=1.0,
        initial_pulls=2,
        trust_alpha=0.05,
        lambda_trust=5.0,
        trust_mode="global",
        w_min=0.01,
        prior_clip_min=0.0,
        prior_clip_max=1.0,
    )
    algo = AdaptiveTrustResidualLinUCB("at_residual_linucb", h)
    # 统一标记：运行循环据此在每轮注入 prior_scores。
    algo.requires_prior_scores = True
    return algo


def _build_prior_provider(args, dataset_name: str, num_actions: int):
    # 运行期配置（例如 prompt dump 数量、debug 输出目录等）
    cfg = BASELINE_RUNTIME_CONFIG
    # prompt_style 决定 prompt_registry 使用哪种模板风格
    prompt_style = getattr(args, "prior_prompt_style", "auto")
    # 为当前数据集构建动作文本（供 prompt 里展示候选动作）
    action_texts = get_action_texts(dataset_name, num_actions, prompt_style=prompt_style)

    # 未启用 LLM prior：直接返回均匀先验（每个动作概率相同）
    if not getattr(args, "use_llm_prior", False):
        provider = UniformPriorProvider(dataset_name=dataset_name)
        return provider, action_texts, "uniform"

    # 使用离线预计算先验（从文件中读取）
    if args.prior_source == "precomputed":
        provider = PrecomputedPriorProvider(
            dataset_name=dataset_name,
            prior_path=args.precomputed_prior_path,
        )
        return provider, action_texts, "precomputed"

    # local_llm 分支：按优先级解析模型来源
    # 优先级：--llm_model > --custom_model > --model(别名)
    model_ref = getattr(args, "llm_model", None)
    if not model_ref:
        # backward compatibility for older scripts
        model_ref = getattr(args, "llm_model_path", None)

    model_path = PRESET_LLM_MODELS.get(model_ref, model_ref) if model_ref else None
    if not model_path:
        model_path = getattr(args, "custom_model", None)
    if not model_path:
        model_alias = getattr(args, "model_name", None)
        if model_alias:
            # 若命中预设别名，映射到真实模型路径/模型ID；否则按原值透传
            model_path = PRESET_LLM_MODELS.get(model_alias, model_alias)

    # 以上都没提供则报错，提示用户至少提供一种模型配置
    if not model_path:
        raise ValueError(
            "No model configured. Use --model/--custom_model or --llm_model"
        )

    # 缓存控制：允许通过开关禁用“跨运行缓存复用”
    cache_dir = getattr(args, "llm_cache_dir", os.path.join(RESULTS_DIR, "prior_cache"))
    rebuild_cache = getattr(args, "rebuild_prior_cache", False)
    if getattr(args, "disable_prior_cache", False):
        run_tag = str(cfg.get("prior_run_tag") or datetime.now().strftime("%Y%m%d_%H%M%S"))
        cache_dir = os.path.join(RESULTS_DIR, "prior_cache_disabled", run_tag)
        rebuild_cache = True

    provider = LocalLLMPriorProvider(
        dataset_name=dataset_name,
        model_path=model_path,
        tokenizer_path=getattr(args, "llm_tokenizer_path", None),
        device=getattr(args, "llm_device", "auto"),
        dtype=getattr(args, "llm_dtype", "auto"),
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
        prompt_style=prompt_style,
        dump_prompts=int(cfg.get("prior_dump_prompts", 0)),
        debug_dir=str(cfg.get("prior_debug_dir", os.path.join(RESULTS_DIR, "prior_debug"))),
        run_tag=cfg.get("prior_run_tag"),
    )
    return provider, action_texts, "local_llm"


register_baseline("epsilon_greedy", _b_epsilon_greedy)
register_baseline("ucb1", _b_ucb1)
register_baseline("linucb", _b_linucb)
register_baseline("neural_ucb", _b_neural_ucb)
register_baseline("neural_linucb", _b_neural_linucb)
# register_baseline("neural_bandit", _b_neural_bandit)
register_baseline("neural_linear", _b_neural_linear)
register_baseline("at_residual_linucb", _b_at_residual_linucb)
# register_baseline("inlucb", _b_inlucb)

BASELINE_NAMES = list(BASELINE_REGISTRY.keys())

# Keep alias names aligned with OurMethod model selection.
PRESET_LLM_MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "qwen2_5_7b": "/home/csg/Awesome-contextual-bandits/models/huggingface/Qwen2.5-7B-Instruct",
    "llama3_1_8b": "meta-llama/Llama-3.1-8B-Instruct",
}


# ====================================================================== #
#  IO helpers                                                             #
# ====================================================================== #

def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def _ourmethod_stem(model_name: str) -> str:
    return f"ourmethod__{_sanitize_name(model_name)}"


def _exp_dir(dataset_name: str, n_rounds: int, seed: int, timestamp: str | None = None) -> str:
    if timestamp:
        folder = f"{timestamp}_{dataset_name}_{n_rounds}r_seed{seed}"
    else:
        folder = f"{dataset_name}_{n_rounds}r_seed{seed}"
    d = os.path.join(RESULTS_DIR, folder)
    os.makedirs(d, exist_ok=True)
    return d


def _find_all_exp_dirs(dataset_name: str, n_rounds: int, seed: int) -> List[str]:
    """返回所有与该实验参数匹配的结果子目录（新格式含时间戳前缀 + 旧格式兼容）。"""
    suffix = f"{dataset_name}_{n_rounds}r_seed{seed}"
    dirs = glob.glob(os.path.join(RESULTS_DIR, f"*_{suffix}"))
    # 兼容旧格式（无时间戳前缀）
    exact = os.path.join(RESULTS_DIR, suffix)
    if os.path.isdir(exact):
        dirs.append(exact)
    return dirs


def save_algo_log(dataset_name: str, algo_stem: str, data: dict, n_rounds: int, seed: int, meta: Dict[str, Any] | None = None, timestamp: str | None = None):
    exp_d = _exp_dir(dataset_name, n_rounds, seed, timestamp=timestamp)
    fp = os.path.join(exp_d, f"{algo_stem}.npz")

    payload = {
        "actions": data["actions"],
        "rewards": data["rewards"],
        "opt_rewards": data["opt_rewards"],
        "opt_actions": data["opt_actions"],
        "step_regret": data["step_regret"],
        "cumulative_regret": data["cumulative_regret"],
        "cumulative_reward": data["cumulative_reward"],
        "__meta_json": json.dumps(meta or {}, ensure_ascii=False),
    }
    
    if "history" in data:
        # Extract detailed UCB timelines for later plotting
        history = data["history"]
        payload["history_min_source"] = np.array([getattr(rec, "min_source", "unknown") for rec in history], dtype=str)
        
        # Safely extract lists, replacing None with empty lists or NaNs to avoid inhomogeneous shape errors
        def safe_extract(rec, attr_name):
            val = getattr(rec, attr_name, None)
            return val if val is not None else []
            
        # Due to numpy's strictness about ragged arrays, we'll store lists of lists as object arrays
        # if dimensions somehow differ, but typically they should all be length K. 
        # Using a list comprehension and explicitly setting dtype=object prevents the ValueError if some are empty.
        s1_vals = [safe_extract(rec, "ucb_s1") for rec in history]
        s2_vals = [safe_extract(rec, "ucb_s2") for rec in history]
        s3_vals = [safe_extract(rec, "ucb_s3") for rec in history]
        
        # Ensure rectangular shape if possible. If some are [], fill with NaN of correct length
        max_lens = max([len(v) for v in s1_vals + s2_vals + s3_vals] + [1])
        s1_vals = [v if len(v) > 0 else [np.nan] * max_lens for v in s1_vals]
        s2_vals = [v if len(v) > 0 else [np.nan] * max_lens for v in s2_vals]
        s3_vals = [v if len(v) > 0 else [np.nan] * max_lens for v in s3_vals]

        payload["history_ucb_s1"] = np.array(s1_vals, dtype=float)
        payload["history_ucb_s2"] = np.array(s2_vals, dtype=float)
        payload["history_ucb_s3"] = np.array(s3_vals, dtype=float)

    np.savez_compressed(fp, **payload)
    print(f"  Saved: {fp}")
    return fp


def load_algo_log(dataset_name: str, algo_stem: str, n_rounds: int, seed: int) -> dict | None:
    # 在所有匹配该实验参数的目录下搜索（支持新格式时间戳前缀和旧格式）
    candidate_dirs = _find_all_exp_dirs(dataset_name, n_rounds, seed)
    matches = []
    for d in candidate_dirs:
        matches.extend(glob.glob(os.path.join(d, f"*_{algo_stem}.npz")))
        fp_exact = os.path.join(d, f"{algo_stem}.npz")
        if os.path.exists(fp_exact):
            matches.append(fp_exact)

    if not matches:
        return None

    # 取修改时间最新的文件
    fp = sorted(matches, key=os.path.getmtime)[-1]
    
    d = np.load(fp, allow_pickle=True)
    out = {k: d[k] for k in d.files if k != "__meta_json"}
    meta = {}
    if "__meta_json" in d.files:
        try:
            meta = json.loads(str(d["__meta_json"]))
        except Exception:
            meta = {}
    out["__meta__"] = meta
    out["__file__"] = fp
    return out


# ====================================================================== #
#  Run one baseline                                                       #
# ====================================================================== #

def run_one_baseline(
    algo,
    algo_name,
    cmab,
    offset,
    n_rounds,
    opt_rewards,
    opt_actions,
    prior_provider=None,
    action_texts=None,
):
    actions = np.zeros(n_rounds, dtype=np.int64)
    rewards = np.zeros(n_rounds)
    t0 = time.time()
    log_freq = max(1, n_rounds // 20)

    for i in range(n_rounds):
        raw_ctx = cmab.context(offset + i)
        ctx = raw_ctx

        if getattr(algo, "requires_prior_scores", False):
            if prior_provider is None or action_texts is None:
                raise ValueError(
                    "Algorithm {} requires prior provider but it is missing".format(algo_name)
                )

            row_id = int(cmab.order[offset + i]) if hasattr(cmab, "order") else int(offset + i)
            context_text = None
            texts = getattr(cmab, "texts", None)
            if texts is not None and row_id < len(texts):
                context_text = str(texts[row_id])

            prior_scores = prior_provider.get_prior_scores(
                sample_id=row_id,
                context=raw_ctx,
                action_texts=action_texts,
                context_text=context_text,
            )
            ctx = {
                "context": raw_ctx,
                "prior_scores": prior_scores,
            }

        a = algo.action(ctx)
        r = float(cmab.reward(offset + i, a))
        algo.update(ctx, a, r)
        actions[i] = a
        rewards[i] = r
        if i % log_freq == 0:
            cr = np.sum(opt_rewards[:i + 1] - rewards[:i + 1])
            print(f"  [{algo_name}] step {i:>5d}/{n_rounds}  cum_regret={cr:.1f}")

    elapsed = time.time() - t0
    step_reg = opt_rewards[:n_rounds] - rewards
    cum_reg = np.cumsum(step_reg)
    cum_rew = np.cumsum(rewards)
    print(
        f"  [{algo_name}] done — cum_regret={cum_reg[-1]:.1f}  "
        f"avg_reward={np.mean(rewards):.4f}  time={elapsed:.1f}s"
    )
    return {
        "actions": actions,
        "rewards": rewards,
        "opt_rewards": opt_rewards[:n_rounds],
        "opt_actions": opt_actions[:n_rounds],
        "step_regret": step_reg,
        "cumulative_regret": cum_reg,
        "cumulative_reward": cum_rew,
    }


# ====================================================================== #
#  Plot helpers                                                           #
# ====================================================================== #

STYLES = {
    "epsilon_greedy": {"color": "#9467bd", "ls": "--", "lw": 1.2},
    "ucb1": {"color": "#2ca02c", "ls": "--", "lw": 1.2},
    "linucb": {"color": "#d62728", "ls": "--", "lw": 1.2},
    "neural_ucb": {"color": "#8c564b", "ls": "-.", "lw": 1.5},
    "neural_linucb": {"color": "#e377c2", "ls": "-.", "lw": 1.5},
    "neural_bandit": {"color": "#1f77b4", "ls": "-", "lw": 1.5},
    "neural_linear": {"color": "#ff7f0e", "ls": "-", "lw": 1.5},
    "at_residual_linucb": {"color": "#7f7f7f", "ls": "-", "lw": 1.8},
    "inlucb": {"color": "#bcbd22", "ls": "-", "lw": 2.0},
    "combined_ucb": {"color": "#17becf", "ls": "-", "lw": 2.5},
}


def _style_for(stem: str) -> Dict[str, Any]:
    if stem.startswith("ourmethod__"):
        return STYLES["combined_ucb"]
    return STYLES.get(stem, {"color": "gray", "ls": "-", "lw": 1.3})


def ensure_plot_compatible(logs: Dict[str, dict]):
    names = list(logs.keys())
    if len(names) <= 1:
        return

    base = logs[names[0]]
    base_n = len(base["rewards"])
    base_opt = base["opt_rewards"]

    for name in names[1:]:
        cur = logs[name]
        if len(cur["rewards"]) != base_n:
            raise ValueError(f"Length mismatch: {names[0]}={base_n}, {name}={len(cur['rewards'])}")
        if not np.array_equal(cur["opt_rewards"], base_opt):
            raise ValueError(
                f"opt_rewards mismatch between {names[0]} and {name}; cannot safely overlay regret curves."
            )


# def print_summary(all_data: Dict[str, dict], n_rounds: int):
#     print(f"\n{'=' * 78}")
#     print(f"{'Algorithm':<28s} {'Cum.Regret':>12s} {'Avg.Reward':>12s} {'Accuracy':>10s}")
#     print(f"{'=' * 78}")
#     for name, data in all_data.items():
#         cr = data["cumulative_regret"][-1]
#         avg_r = np.mean(data["rewards"])
#         acc = np.mean(data["actions"] == data["opt_actions"])
#         print(f"{name:<28s} {cr:>12.1f} {avg_r:>12.4f} {acc:>10.4f}")
#     print(f"{'=' * 78}")

def print_summary(all_data: Dict[str, dict], n_rounds: int):
    print(f"\n{'=' * 120}")
    print(f"{'Algorithm':<20s} {'Cum.Regret':>12s} {'Avg.Reward':>12s} {'Accuracy':>10s}  Params")
    print(f"{'=' * 120}")

    for name, data in all_data.items():
        cr = data["cumulative_regret"][-1]
        avg_r = np.mean(data["rewards"])
        acc = np.mean(data["actions"] == data["opt_actions"])

        # ✅ 取参数
        hparams = data.get("hparams", {})
        
        # 只打印关键参数（避免太长）
        if isinstance(hparams, dict):
            key_params = {
                k: hparams[k]
                for k in ["initial_lr", "lambda_prior", "alpha", "training_epochs"]
                if k in hparams
            }
        else:
            key_params = {}

        print(f"{name:<20s} {cr:>12.1f} {avg_r:>12.4f} {acc:>10.4f}  {key_params}")

    print(f"{'=' * 120}")

def plot_cumulative_reward(dataset_name: str, all_data: Dict[str, dict], n_rounds: int, seed: int, tag: str, save_rounds: int = None, exp_d: str = None):
    ensure_plot_compatible(all_data)

    # Use provided exp_d, or fall back to deriving from save_rounds/n_rounds
    if exp_d is None:
        exp_d = _exp_dir(dataset_name, save_rounds if save_rounds else n_rounds, seed)
    steps = np.arange(1, n_rounds + 1)
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, data in all_data.items():
        s = _style_for(name)
        # 逐点绘制累积奖励，长度固定为 n_rounds
        ax.plot(steps, data["cumulative_reward"], label=name, **s)

    ax.set_xlabel("Time Step", fontsize=13)
    ax.set_ylabel("Cumulative Reward", fontsize=13)
    ax.set_title(f"{dataset_name} — Cumulative Reward ({n_rounds}r, seed={seed})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fp = os.path.join(exp_d, f"{tag}_cum_reward.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {fp}")
    return fp


def plot_average_reward(dataset_name: str, all_data: Dict[str, dict], n_rounds: int, seed: int, tag: str, save_rounds: int = None, exp_d: str = None):

    ensure_plot_compatible(all_data)
    if exp_d is None:
        exp_d = _exp_dir(dataset_name, save_rounds if save_rounds else n_rounds, seed)
    steps = np.arange(1, n_rounds + 1)
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, data in all_data.items():
        s = _style_for(name)
        # 同样逐点绘制平均奖励 (平滑版: 当前累积/时间步)
        avg_rewards = data["cumulative_reward"] / steps
        ax.plot(steps, avg_rewards, label=name, **s)
    ax.set_xlabel("Time Step", fontsize=13)
    ax.set_ylabel("Cumulative Average Reward", fontsize=13)
    ax.set_title(f"{dataset_name} — Average Reward ({n_rounds}r, seed={seed})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(exp_d, f"{tag}_avg_reward.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {fp}")
    return fp

def plot_regret(dataset_name: str, all_data: Dict[str, dict], n_rounds: int, seed: int, tag: str, save_rounds: int = None, exp_d: str = None):
    ensure_plot_compatible(all_data)

    if exp_d is None:
        exp_d = _exp_dir(dataset_name, save_rounds if save_rounds else n_rounds, seed)
    steps = np.arange(1, n_rounds + 1)
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, data in all_data.items():
        s = _style_for(name)
        ax.plot(steps, data["cumulative_regret"], label=name, **s)

    ax.set_xlabel("Time Step", fontsize=13)
    ax.set_ylabel("Cumulative Regret", fontsize=13)
    ax.set_title(f"{dataset_name} — Cumulative Regret ({n_rounds}r, seed={seed})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fp = os.path.join(exp_d, f"{tag}_regret.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {fp}")
    return fp


# ====================================================================== #
#  Selectors                                                              #
# ====================================================================== #

def _resolve_algorithm_selectors(selectors: List[str]) -> List[str]:
    resolved: List[str] = []

    for item in selectors:
        if item == "all_baselines":
            resolved.extend(BASELINE_NAMES)
            continue
        if item in BASELINE_NAMES:
            resolved.append(item)
            continue
        if item.startswith("ourmethod:"):
            model = item.split(":", 1)[1].strip()
            if not model:
                raise ValueError("Selector 'ourmethod:' requires model name, e.g. ourmethod:smollm2")
            resolved.append(_ourmethod_stem(model))
            continue
        raise ValueError(
            f"Unknown selector: {item}. Use baseline name, all_baselines, or ourmethod:<model>."
        )

    uniq = []
    seen = set()
    for r in resolved:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq


# ====================================================================== #
#  Commands                                                               #
# ====================================================================== #

def cmd_baselines(args):
    if getattr(args, "llm_model", None) and not getattr(args, "use_llm_prior", False):
        # If user explicitly specifies an LLM model, default to enabling LLM prior.
        args.use_llm_prior = True
        print("[INFO] --llm_model detected; auto enabling --use_llm_prior")

    global BASELINE_RUNTIME_CONFIG
    BASELINE_RUNTIME_CONFIG = {
        "prior_dump_prompts": getattr(args, "prior_dump_prompts", 0),
        "prior_debug_dir": getattr(args, "prior_debug_dir", os.path.join(RESULTS_DIR, "prior_debug")),
    }

    datasets = [d.lower() for d in args.datasets]
    baselines = _resolve_algorithm_selectors(args.baselines)
    baselines = [b for b in baselines if b in BASELINE_NAMES]

    for dname in datasets:
        if dname not in DATASET_REGISTRY:
            print(f"[ERROR] Unsupported dataset: {dname}")
            continue

        n_rounds = args.n_rounds
        seed = args.seed
        cold_start_n = args.cold_start_n

        cmab, opt_rewards, opt_actions, num_actions, context_dim, n_rounds = DATASET_REGISTRY[dname](
            n_rounds, cold_start_n, seed
        )

        print(f"=== Baselines on {dname} ===")
        print(f"  n_rounds={n_rounds}, seed={seed}, cold_start_n={cold_start_n}")
        print(f"  context_dim={context_dim}, num_actions={num_actions}")

        all_data: Dict[str, dict] = {}
        # 为当前 run 生成唯一的时间戳，供新建保存时使用
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        BASELINE_RUNTIME_CONFIG["prior_run_tag"] = run_timestamp

        for bname in baselines:
            # 当禁用 prior cache 时，必须重跑，避免误复用旧实验结果导致未触发 LLM prior。
            should_skip_existing = args.force or getattr(args, "disable_prior_cache", False)
            existing = None if should_skip_existing else load_algo_log(dname, bname, n_rounds, seed)
            if existing is not None:
                print(f"  [SKIP] {bname} (loaded existing)")
                all_data[bname] = existing
                continue

            print(f"  Running {bname} ...")
            t_start = time.time()
            _set_all_seeds(seed)
            algo = BASELINE_REGISTRY[bname](num_actions, context_dim)

            prior_provider = None
            action_texts = None
            prior_backend = "none"
            prompt_template = None
            if getattr(algo, "requires_prior_scores", False):
                prompt_template = get_dataset_prompt_template_name(dname)
                try:
                    prior_provider, action_texts, prior_backend = _build_prior_provider(
                        args=args,
                        dataset_name=dname,
                        num_actions=num_actions,
                    )
                    print(
                        f"    prior backend={prior_backend}, prompt_template={prompt_template}, "
                        f"cache_dir={getattr(args, 'llm_cache_dir', os.path.join(RESULTS_DIR, 'prior_cache'))}"
                    )
                    if prior_backend == "local_llm" and getattr(args, "prior_dump_prompts", 0) > 0:
                        print(
                            f"    prior debug dump={getattr(args, 'prior_dump_prompts', 0)}, "
                            f"output_dir={os.path.join(getattr(args, 'prior_debug_dir', os.path.join(RESULTS_DIR, 'prior_debug')), dname)}"
                        )
                except Exception as e:
                    print(f"    [WARN] prior init failed ({e}); fallback to uniform prior")
                    prior_provider = UniformPriorProvider(dataset_name=dname)
                    action_texts = get_action_texts(dname, num_actions, prompt_style=getattr(args, "prior_prompt_style", "auto"))
                    prior_backend = "uniform_fallback"

            # result = run_one_baseline(
            #     algo, bname, cmab, offset=cold_start_n,
            #     n_rounds=n_rounds, opt_rewards=opt_rewards, opt_actions=opt_actions,
            # )

            # ✅ 记录超参数（关键！！）
            if hasattr(algo, "hparams"):
                result_hparams = algo.hparams
            else:
                result_hparams = vars(algo) if hasattr(algo, "__dict__") else {}

            result = run_one_baseline(
                algo, bname, cmab, offset=cold_start_n,
                n_rounds=n_rounds, opt_rewards=opt_rewards, opt_actions=opt_actions,
                prior_provider=prior_provider,
                action_texts=action_texts,
            )
            result["hparams"] = result_hparams



            t_total = time.time() - t_start
            result["times"] = {"total": t_total}
            
            meta = {
                "dataset": dname,
                "seed": seed,
                "n_rounds": n_rounds,
                "cold_start_n": cold_start_n,
                "algorithm": bname,
                "type": "baseline",
                "prior_backend": prior_backend,
                "prompt_template": prompt_template,
                "saved_at": datetime.now().isoformat(),
                **vars(args),
            }
            # 使用时间戳作为命名前缀
            record_stem = f"{run_timestamp}_{bname}"
            save_algo_log(dname, record_stem, result, n_rounds, seed, meta=meta, timestamp=run_timestamp)

            # 自动记录单算法实验日志（基础信息、指标、逐轮 CSV、Markdown 模板）
            try:
                run_exp_d = _exp_dir(dname, n_rounds, seed, timestamp=run_timestamp)
                log_dir = save_single_run_log(
                    exp_dir=run_exp_d,
                    dataset_name=dname,
                    algorithm_name=bname,
                    args=vars(args),
                    meta=meta,
                    result=result,
                    prompt_info={
                        "prior_backend": prior_backend,
                        "prompt_template": prompt_template,
                        "prior_prompt_style": getattr(args, "prior_prompt_style", "auto"),
                        "action_text_count": len(action_texts) if action_texts is not None else 0,
                    },
                    plot_paths=None,
                )
                print(f"  Auto experiment log saved: {log_dir}")
            except Exception as e:
                print(f"  [WARN] Failed to save auto experiment log for {bname}: {e}")

            all_data[bname] = result
            



        if all_data:
            print_summary(all_data, n_rounds)
            tag = f"{run_timestamp}_baselines_" + _sanitize_name("_".join(baselines))
            run_exp_d = _exp_dir(dname, n_rounds, seed, timestamp=run_timestamp)
            regret_plot_fp = plot_regret(dname, all_data, n_rounds, seed, tag=tag, exp_d=run_exp_d)
            cum_reward_plot_fp = plot_cumulative_reward(dname, all_data, n_rounds, seed, tag=tag, exp_d=run_exp_d)
            avg_reward_plot_fp = plot_average_reward(dname, all_data, n_rounds, seed, tag=tag, exp_d=run_exp_d)

            def _to_json_safe(v):
                """将不可序列化的对象（如 argparse.Namespace）转为可序列化格式。"""
                import argparse
                if isinstance(v, argparse.Namespace):
                    return vars(v)
                if isinstance(v, dict):
                    return {k: _to_json_safe(val) for k, val in v.items()}
                if isinstance(v, (list, tuple)):
                    return [_to_json_safe(i) for i in v]
                return v

            safe_args = {k: _to_json_safe(v) for k, v in vars(args).items()}
            summary = {
                "meta": {
                    "dataset": dname,
                    "n_rounds": n_rounds,
                    "seed": seed,
                    "cold_start_n": cold_start_n,
                    "baselines": baselines,
                    **safe_args,
                },
                "results": {
                    name: {
                        "cumulative_regret": float(data["cumulative_regret"][-1]),
                        "average_reward": float(np.mean(data["rewards"])),
                        "accuracy": float(np.mean(data["actions"] == data["opt_actions"])),
                        "times": data.get("times", {}),
                        "hparams": _to_json_safe(data.get("hparams", {})),
                    }
                    for name, data in all_data.items()
                },
            }
            fp = os.path.join(run_exp_d, f"{run_timestamp}_baselines_summary.json")
            with open(fp, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"  Summary saved: {fp}")

            # 自动记录数据集级实验总结日志，便于后续补充手写分析
            try:
                summary_results = {
                    name: {
                        "final_cumulative_regret": float(data["cumulative_regret"][-1]),
                        "average_reward": float(np.mean(data["rewards"])),
                        "accuracy": float(np.mean(data["actions"] == data["opt_actions"])),
                    }
                    for name, data in all_data.items()
                }
                summary_log_dir = save_dataset_summary_log(
                    exp_dir=run_exp_d,
                    dataset_name=dname,
                    summary_name=f"{run_timestamp}_baselines_summary",
                    args=vars(args),
                    summary_meta=summary["meta"],
                    summary_results=summary_results,
                    plot_paths={
                        "regret_plot": regret_plot_fp,
                        "cumulative_reward_plot": cum_reward_plot_fp,
                        "average_reward_plot": avg_reward_plot_fp,
                    },
                )
                print(f"  Auto dataset summary log saved: {summary_log_dir}")
            except Exception as e:
                print(f"  [WARN] Failed to save dataset summary log: {e}")


def cmd_ourmethod(args):
    from OurMethod.run_full_experiment import run_our_method_pipeline
    from OurMethod.core.protocol import Arm
    from OurMethod.core.prompt_builder import StructuredPromptBuilder

    model_for_name = args.custom_model if args.custom_model else args.model_name
    our_stem = _ourmethod_stem(model_for_name)
    if hasattr(args, "ablation_mode") and args.ablation_mode != "none":
        our_stem += f"_{args.ablation_mode}"

    for dname in [d.lower() for d in args.datasets]:
        if dname not in DATASET_REGISTRY:
            print(f"[ERROR] Unsupported dataset: {dname}")
            continue

        n_rounds = args.n_rounds
        seed = args.seed
        cold_start_n = args.cold_start_n

        cmab, opt_rewards, opt_actions, num_actions, context_dim, n_rounds = DATASET_REGISTRY[dname](
            n_rounds, cold_start_n, seed
        )

        print(f"=== OurMethod on {dname} ===")
        print(f"  model={model_for_name}, n_rounds={n_rounds}, seed={seed}, cold_start_n={cold_start_n}")

        custom_arms = None
        custom_pb = None
        if dname == "newsgroups":
            categories = [
                ("comp.graphics", "Computer graphics discussions"),
                ("rec.sport.baseball", "Baseball sports discussions"),
                ("sci.med", "Medical science discussions"),
                ("sci.space", "Space science discussions"),
                ("talk.politics.guns", "Gun politics discussions"),
                ("soc.religion.christian", "Christian religion discussions"),
            ]
            custom_arms = [Arm(arm_id=i, name=n, description=desc) for i, (n, desc) in enumerate(categories)]
            custom_pb = StructuredPromptBuilder(
                role="text topic classification expert",
                scenario=(
                    "20 Newsgroups topic classification as contextual bandit. "
                    "Given TF-IDF/LSA semantic features of a document, predict which newsgroup topic it belongs to."
                ),
                max_feedback=5,
                max_feature_display=10,
            )
        elif dname == "ag_news":
            categories = [
                ("World", "international politics and global events"),
                ("Sports", "sports competitions and athletes"),
                ("Business", "economy, finance, companies"),
                ("Sci/Tech", "science and technology innovations"),
            ]
            custom_arms = [Arm(arm_id=i, name=n, description=desc) for i, (n, desc) in enumerate(categories)]
            custom_pb = StructuredPromptBuilder(
                role="news topic classification expert",
                scenario=(
                    "AG News topic classification as contextual bandit. "
                    "Given the text of a news article, predict which topic it belongs to."
                ),
                max_feedback=5,
                max_feature_display=0,
            )
        elif dname == "statlog":
            custom_arms = [
                Arm(arm_id=i, name=f"state_{i + 1}",
                    description=f"Shuttle operating state class {i + 1}")
                for i in range(num_actions)
            ]
            custom_pb = StructuredPromptBuilder(
                role="shuttle state classification expert",
                scenario=(
                    "Statlog Shuttle state classification as contextual bandit. "
                    "Given sensor-derived numeric features, predict the most likely shuttle state class."
                ),
                max_feedback=5,
                max_feature_display=9,
            )
        elif dname == "magic":
            categories = [
                ("gamma", "gamma-ray event"),
                ("hadron", "hadron background event"),
            ]
            custom_arms = [Arm(arm_id=i, name=n, description=desc) for i, (n, desc) in enumerate(categories)]
            custom_pb = StructuredPromptBuilder(
                role="gamma telescope event classification expert",
                scenario=(
                    "MAGIC Gamma Telescope event classification as contextual bandit. "
                    "Given physics-inspired image features, distinguish gamma-ray events from hadron background."
                ),
                max_feedback=5,
                max_feature_display=11,
            )
        elif dname == "covertype":
            custom_arms = [
                Arm(arm_id=i, name=f"cover_type_{i + 1}",
                    description=f"Forest cover type class {i + 1}")
                for i in range(num_actions)
            ]
            custom_pb = StructuredPromptBuilder(
                role="forest cover type classification expert",
                scenario=(
                    "Covertype classification as contextual bandit. "
                    "Given geospatial and environmental features, predict the forest cover type."
                ),
                max_feedback=5,
                max_feature_display=20,
            )
        elif dname == "mnist":
            custom_arms = [
                Arm(arm_id=i, name=f"digit_{i}", description=f"Handwritten digit {i}")
                for i in range(num_actions)
            ]
            custom_pb = StructuredPromptBuilder(
                role="handwritten digit classification expert",
                scenario=(
                    "MNIST digit classification as contextual bandit. "
                    "Given pixel-intensity features of a digit image, predict the digit class 0-9."
                ),
                max_feedback=5,
                max_feature_display=32,
            )
        elif dname == "adult":
            if num_actions == 2:
                categories = [
                    ("income_low", "income class (typically <=50K)"),
                    ("income_high", "income class (typically >50K)"),
                ]
                custom_arms = [Arm(arm_id=i, name=n, description=desc) for i, (n, desc) in enumerate(categories)]
            else:
                custom_arms = [
                    Arm(arm_id=i, name=f"income_class_{i}", description=f"Income-related class {i}")
                    for i in range(num_actions)
                ]
            custom_pb = StructuredPromptBuilder(
                role="demographic income classification expert",
                scenario=(
                    "Adult income classification as contextual bandit. "
                    "Given demographic and employment features, predict the income category."
                ),
                max_feedback=5,
                max_feature_display=24,
            )
        elif dname in ("census", "cencus"):
            custom_arms = [
                Arm(arm_id=i, name=f"census_class_{i}", description=f"Census target class {i}")
                for i in range(num_actions)
            ]
            custom_pb = StructuredPromptBuilder(
                role="census category classification expert",
                scenario=(
                    "US Census classification as contextual bandit. "
                    "Given census demographic and socio-economic features, predict the target category."
                ),
                max_feedback=5,
                max_feature_display=24,
            )

        _set_all_seeds(seed)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result = run_our_method_pipeline(
            cmab=cmab,
            opt_rewards=opt_rewards,
            opt_actions=opt_actions,
            n_rounds=n_rounds,
            num_actions=num_actions,
            context_dim=context_dim,
            cold_start_n=cold_start_n,
            model_name=args.model_name,
            custom_model=args.custom_model,
            seed=seed,
            z_dim=args.z_dim,
            c_ucb1=args.c_ucb1,
            alpha_linucb=args.alpha_linucb,
            c_sim=args.c_sim,
            offline_freq=args.offline_freq,
            offline_epochs=args.offline_epochs,
            warmup_epochs=args.warmup_epochs,
            verbose=True,
            dataset_name=dname,
            no_cache=args.no_cache,
            arms=custom_arms,
            prompt_builder=custom_pb,
            dump_prompts=args.dump_prompts,
            ablation_mode=getattr(args, "ablation_mode", "none"),
            run_timestamp=run_timestamp,
        )

        if not result:
            print("[ERROR] OurMethod returned empty result.")
            continue

        meta = {
            "dataset": dname,
            "seed": seed,
            "n_rounds": n_rounds,
            "cold_start_n": cold_start_n,
            "algorithm": our_stem,
            "type": "ourmethod",
            "model_name": args.model_name,
            "custom_model": args.custom_model,
            "saved_at": datetime.now().isoformat(),
            **vars(args),
        }
        record_stem = f"{run_timestamp}_{our_stem}"

        save_algo_log(dname, record_stem, result, n_rounds, seed, meta=meta, timestamp=run_timestamp)

        # 自动记录 OurMethod 单次实验日志
        try:
            run_exp_d = _exp_dir(dname, n_rounds, seed, timestamp=run_timestamp)
            log_dir = save_single_run_log(
                exp_dir=run_exp_d,
                dataset_name=dname,
                algorithm_name=our_stem,
                args=vars(args),
                meta=meta,
                result=result,
                prompt_info={
                    "model_for_name": model_for_name,
                    "ablation_mode": getattr(args, "ablation_mode", "none"),
                },
                plot_paths=None,
            )
            print(f"  Auto experiment log saved: {log_dir}")
        except Exception as e:
            print(f"  [WARN] Failed to save auto experiment log for OurMethod: {e}")

        print(
            f"  OurMethod[{model_for_name}] cum_regret={result['cumulative_regret'][-1]:.1f}  "
            f"avg_reward={np.mean(result['rewards']):.4f}"
        )

        # Generate separate OurMethod Summary Report
        try:
            exp_d = _exp_dir(dname, n_rounds, seed, timestamp=run_timestamp)
            report_path = os.path.join(exp_d, f"{record_stem}_run_report.md")
            history = result.get("history", [])
            if history:
                sources = [getattr(rec, "min_source", "unknown") for rec in history]
                if sources:
                    src_counts = {s: sources.count(s) for s in set(sources)}
                    
                    with open(report_path, "w", encoding='utf-8') as f:
                        f.write(f"# OurMethod 运行深度总结报告 ({our_stem})\n\n")
                        f.write(f"- **数据集**: {dname}\n")
                        f.write(f"- **总执行步数**: {len(history)}\n")
                        f.write(f"- **累计遗憾 (Cum. Regret)**: {result['cumulative_regret'][-1]:.2f}\n")
                        f.write(f"- **平均奖励 (Avg. Reward)**: {np.mean(result['rewards']):.4f}\n")
                        f.write(f"- **分类准确率 (Accuracy)**: {np.mean(result['actions'] == result['opt_actions']):.4f}\n\n")
                        
                        f.write("## 1. 决策来源分布 (Selection Source Distribution)\n\n")
                        f.write("| 策略源 | 标识 | 选中次数 | 选中比例 |\n")
                        f.write("| :--- | :---: | :---: | :---: |\n")
                        for s, count in src_counts.items():
                            name = {"s1": "LinUCB", "s2": "UCB1", "s3": "LLM-based"}.get(s, s)
                            f.write(f"| {name} | {s} | {count} | {count/len(history)*100:.2f}% |\n")
                            
                        f.write("\n## 2. 三种平行策略的 UCB 分值与置信度深度监控 (Mean Stats)\n\n")
                        f.write("| 策略维度 | S1 (LinUCB) | S2 (UCB1) | S3 (LLM) |\n")
                        f.write("| :--- | :---: | :---: | :---: |\n")
                        
                        # 获取三方数据的辅助计算 (全局)
                        def get_overall_avg(key):
                            attrs = {
                                "ucb_s1": "ucb_s1", "ucb_s2": "ucb_s2", "ucb_s3": "ucb_s3",
                                "radius_linucb": "radius_linucb", "radius_ucb1": "radius_ucb1", "radius_llm": "radius_llm"
                            }
                            attr = attrs.get(key, key)
                            vals = [np.mean(getattr(rec, attr)) for rec in history if getattr(rec, attr, None) is not None]
                            return f"{np.mean(vals):.4f}" if vals else "N/A"

                        f.write(f"| **平均 UCB 得分 (Mean Score)** | {get_overall_avg('ucb_s1')} | {get_overall_avg('ucb_s2')} | {get_overall_avg('ucb_s3')} |\n")
                        f.write(f"| **平均置信半径 (Mean Radius)** | {get_overall_avg('radius_linucb')} | {get_overall_avg('radius_ucb1')} | {get_overall_avg('radius_llm')} |\n")

                        # --- 新增：分阶段分析 (Phased Analysis) ---
                        f.write("\n## 3. 分阶段动态表现分析 (Phased Performance Analysis)\n\n")
                        f.write("| 阶段 (Stage) | 选中 S1% | 选中 S2% | 选中 S3% | 平均奖励 | 准确率 |\n")
                        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
                        
                        total_steps = len(history)
                        # 定义阶段：10%, 30%, 50%, 100% (或自定义)
                        phases = [0.1, 0.3, 0.5, 1.0]
                        for p in phases:
                            limit = int(total_steps * p)
                            sub_hist = history[:limit]
                            sub_rewards = result['rewards'][:limit]
                            sub_actions = result['actions'][:limit]
                            sub_opt = result['opt_actions'][:limit]
                            
                            sub_sources = [getattr(rec, "min_source", "unknown") for rec in sub_hist]
                            s1_p = (sub_sources.count("s1") / len(sub_sources) * 100) if sub_sources else 0
                            s2_p = (sub_sources.count("s2") / len(sub_sources) * 100) if sub_sources else 0
                            s3_p = (sub_sources.count("s3") / len(sub_sources) * 100) if sub_sources else 0
                            
                            avg_r = np.mean(sub_rewards) if len(sub_rewards) > 0 else 0
                            acc = np.mean(sub_actions == sub_opt) if len(sub_actions) > 0 else 0
                            
                            f.write(f"| 前 {int(p*100)}% ({limit}步) | {s1_p:.1f}% | {s2_p:.1f}% | {s3_p:.1f}% | {avg_r:.4f} | {acc:.4f} |\n")

                        f.write("\n## 4. 阶段性 UCB 得分走势 (UCB Score Evolution)\n\n")
                        f.write("| 阶段 (Stage) | S1 Mean UCB | S2 Mean UCB | S3 Mean UCB |\n")
                        f.write("| :--- | :---: | :---: | :---: |\n")
                        for p in phases:
                            limit = int(total_steps * p)
                            sub_hist = history[:limit]
                            def get_sub_avg(key):
                                vals = [np.mean(getattr(rec, key)) for rec in sub_hist if getattr(rec, key, None) is not None]
                                return f"{np.mean(vals):.4f}" if vals else "N/A"
                            f.write(f"| 前 {int(p*100)}% | {get_sub_avg('ucb_s1')} | {get_sub_avg('ucb_s2')} | {get_sub_avg('ucb_s3')} |\n")

                        f.write(f"\n- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    print(f"  Detailed OurMethod report saved: {report_path}")
        except Exception as e:
            print(f"  [WARN] Failed to generate detailed report: {e}")

        summary = {
            "meta": meta,
            "results": {
                "cumulative_regret": float(result["cumulative_regret"][-1]),
                "average_reward": float(np.mean(result["rewards"])),
                "accuracy": float(np.mean(result["actions"] == result["opt_actions"])),
                "ucb_dominance": extract_ucb_stats(result["history"]) if "history" in result else {},
                "times": result.get("times", {}),
            },
        }
        run_exp_d = _exp_dir(dname, n_rounds, seed, timestamp=run_timestamp)
        fp = os.path.join(run_exp_d, f"{record_stem}_summary.json")
        with open(fp, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  Summary saved: {fp}")

        # 与 baselines 命令对齐：OurMethod 结束后自动产出至少一张累计遗憾图
        try:
            plot_data = {our_stem: result}
            plot_tag = f"{run_timestamp}_{our_stem}"
            plot_fp = plot_regret(
                dname,
                plot_data,
                n_rounds,
                seed,
                tag=plot_tag,
                exp_d=run_exp_d,
            )
            print(f"  Auto plot saved: {plot_fp}")
        except Exception as e:
            print(f"  [WARN] Failed to generate OurMethod plot: {e}")

        # 自动记录 OurMethod 数据集级 summary 日志（单算法版本）
        try:
            summary_log_dir = save_dataset_summary_log(
                exp_dir=run_exp_d,
                dataset_name=dname,
                summary_name=f"{record_stem}_summary",
                args=vars(args),
                summary_meta=meta,
                summary_results={
                    our_stem: {
                        "final_cumulative_regret": float(result["cumulative_regret"][-1]),
                        "average_reward": float(np.mean(result["rewards"])),
                        "accuracy": float(np.mean(result["actions"] == result["opt_actions"])),
                    }
                },
                plot_paths=None,
            )
            print(f"  Auto dataset summary log saved: {summary_log_dir}")
        except Exception as e:
            print(f"  [WARN] Failed to save dataset summary log for OurMethod: {e}")


def cmd_plot(args):
    selectors = args.algorithms
    try:
        stems = _resolve_algorithm_selectors(selectors)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    # Check if a custom prefix limit was set
    limit_rounds = getattr(args, "limit_rounds", None)

    for dname in [d.lower() for d in args.datasets]:
        if dname not in DATASET_REGISTRY:
            print(f"[ERROR] Unsupported dataset: {dname}")
            continue

        n_rounds = args.n_rounds
        seed = args.seed

        loaded: Dict[str, dict] = {}
        for stem in stems:
            d = load_algo_log(dname, stem, n_rounds, seed)
            if d is None:
                print(f"  [{dname}] [SKIP] missing: {stem}")
                continue
            
            # Apply truncation if limit_rounds is set and strictly smaller than n_rounds
            if limit_rounds is not None and limit_rounds > 0 and limit_rounds < n_rounds:
                max_idx = min(limit_rounds, len(d["rewards"]))
                d["actions"] = d["actions"][:max_idx]
                d["rewards"] = d["rewards"][:max_idx]
                d["opt_rewards"] = d["opt_rewards"][:max_idx]
                d["opt_actions"] = d["opt_actions"][:max_idx]
                d["step_regret"] = d["step_regret"][:max_idx]
                d["cumulative_regret"] = d["cumulative_regret"][:max_idx]
                d["cumulative_reward"] = d["cumulative_reward"][:max_idx]
            
            loaded[stem] = d
            print(f"  [{dname}] loaded: {stem}")

        if not loaded:
            print(f"[ERROR] {dname}: no logs found for selectors {selectors}")
            continue

        try:
            ensure_plot_compatible(loaded)
        except ValueError as e:
            print(f"[ERROR] {dname}: {e}")
            print("       Please rerun selected algorithms with identical dataset/n_rounds/seed.")
            continue

        # Adjust the effective n_rounds for the summary and titles if limited
        actual_plot_rounds = min(n_rounds, limit_rounds) if limit_rounds else n_rounds
        print_summary(loaded, actual_plot_rounds)

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"{run_timestamp}_plot_" + _sanitize_name("_".join(stems))

        # 读取数据所在的目录作为 plot 输出目录，保证 plot 和数据放在一起
        plot_exp_d = os.path.dirname(list(loaded.values())[0].get("__file__", ""))
        if not plot_exp_d or not os.path.isdir(plot_exp_d):
            plot_exp_d = _exp_dir(dname, n_rounds, seed)

        # Pass actual_plot_rounds to limit the x-axis properly, but pass n_rounds for the directory lookup
        plot_regret(dname, loaded, actual_plot_rounds, seed, tag=tag, save_rounds=n_rounds, exp_d=plot_exp_d)
        plot_cumulative_reward(dname, loaded, actual_plot_rounds, seed, tag=tag, save_rounds=n_rounds, exp_d=plot_exp_d)
        plot_average_reward(dname, loaded, actual_plot_rounds, seed, tag=tag, save_rounds=n_rounds, exp_d=plot_exp_d)

        summary = {
            "meta": {
                "dataset": dname,
                "n_rounds": n_rounds,
                "plot_limited_rounds": actual_plot_rounds,
                "seed": seed,
                "selectors": selectors,
                "stems": stems,
            },
            "results": {
                name: {
                    "cumulative_regret": float(data["cumulative_regret"][-1]),
                    "average_reward": float(np.mean(data["rewards"])),
                    "accuracy": float(np.mean(data["actions"] == data["opt_actions"])),
                    "file": data.get("__file__", ""),
                }
                for name, data in loaded.items()
            },
        }
        fp = os.path.join(plot_exp_d, f"{tag}_summary.json")
        with open(fp, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  Plot summary saved: {fp}")


def cmd_full(args):
    """Run baselines + OurMethod + regret plot in one command for each dataset."""
    baseline_args = argparse.Namespace(
        datasets=args.datasets,
        baselines=["all_baselines"],
        n_rounds=args.n_rounds,
        cold_start_n=args.cold_start_n,
        seed=args.seed,
        force=args.force,
        use_llm_prior=args.use_llm_prior,
        prior_source=args.prior_source,
        precomputed_prior_path=args.precomputed_prior_path,
        prior_prompt_style=args.prior_prompt_style,
        llm_model=args.llm_model,
        llm_tokenizer_path=args.llm_tokenizer_path,
        llm_device=args.llm_device,
        llm_dtype=args.llm_dtype,
        llm_cache_dir=args.llm_cache_dir,
        rebuild_prior_cache=args.rebuild_prior_cache,
        disable_prior_cache=args.disable_prior_cache,
        prior_dump_prompts=args.prior_dump_prompts,
        prior_debug_dir=args.prior_debug_dir,
    )
    cmd_baselines(baseline_args)

    our_args = argparse.Namespace(
        datasets=args.datasets,
        n_rounds=args.n_rounds,
        cold_start_n=args.cold_start_n,
        seed=args.seed,
        model_name=args.model_name,
        custom_model=args.custom_model,
        z_dim=args.z_dim,
        c_ucb1=args.c_ucb1,
        alpha_linucb=args.alpha_linucb,
        c_sim=args.c_sim,
        offline_freq=args.offline_freq,
        offline_epochs=args.offline_epochs,
        warmup_epochs=args.warmup_epochs,
        no_cache=args.no_cache,
        dump_prompts=args.dump_prompts,
        ablation_mode=args.ablation_mode,
    )
    cmd_ourmethod(our_args)

    model_for_name = args.custom_model if args.custom_model else args.model_name
    plot_args = argparse.Namespace(
        datasets=args.datasets,
        algorithms=["all_baselines", f"ourmethod:{model_for_name}"],
        n_rounds=args.n_rounds,
        limit_rounds=None,
        seed=args.seed,
    )
    cmd_plot(plot_args)


# ====================================================================== #
#  Main                                                                   #
# ====================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Generic step-by-step experiment runner")
    sub = parser.add_subparsers(dest="command")

    # baselines
    p1 = sub.add_parser("baselines", help="Run selected baselines on selected datasets")
    p1.add_argument("--datasets", nargs="+", default=["statlog_shuttle"], help=f"Supported: {sorted(DATASET_REGISTRY.keys())}")
    p1.add_argument("--baselines", nargs="+", default=["all_baselines"], help="Baseline selectors: all_baselines or subset names")
    p1.add_argument("--n_rounds", type=int, default=10000)
    p1.add_argument("--cold_start_n", type=int, default=0)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--force", action="store_true", help="Ignore existing logs and rerun")

    
    p1.add_argument("--use_llm_prior", action="store_true", help="Enable LLM prior for algorithms that require priors")
    p1.add_argument("--prior_source", type=str, default="local_llm", choices=["local_llm", "precomputed"], help="Prior provider backend")
    p1.add_argument("--precomputed_prior_path", type=str, default=None, help="Path to precomputed priors when prior_source=precomputed")
    p1.add_argument("--prior_prompt_style", type=str, default="auto", help="Prompt style hint used by prompt registry")
    p1.add_argument(
        "--llm_model",
        "--llm_model_path",
        dest="llm_model",
        type=str,
        default=None,
        help="LLM model alias or local model directory for AutoModelForCausalLM",
    )
    p1.add_argument("--llm_tokenizer_path", type=str, default=None, help="Local tokenizer directory (defaults to model path)")
    p1.add_argument("--llm_device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    p1.add_argument("--llm_dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="Inference dtype")
    p1.add_argument("--llm_cache_dir", type=str, default=os.path.join(RESULTS_DIR, "prior_cache"), help="Disk cache directory for priors")
    p1.add_argument("--rebuild_prior_cache", action="store_true", help="Rebuild prior cache before run")
    p1.add_argument("--disable_prior_cache", action="store_true", help="Disable cross-run prior cache reuse")
    p1.add_argument("--prior_dump_prompts", type=int, default=3, help="Dump first N prior prompts to results/prior_debug/<dataset>/")
    p1.add_argument("--prior_debug_dir", type=str, default=os.path.join(RESULTS_DIR, "prior_debug"), help="Debug output directory for dumped prior prompts")


    # ourmethod
    p2 = sub.add_parser("ourmethod", help="Run OurMethod on selected datasets")
    p2.add_argument("--datasets", nargs="+", default=["statlog_shuttle"], help=f"Supported: {sorted(DATASET_REGISTRY.keys())}")
    p2.add_argument("--n_rounds", type=int, default=2000)
    p2.add_argument("--cold_start_n", type=int, default=50)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--model", dest="model_name", type=str, default="smollm2",
                    help="LLM model alias (stub/smollm2/qwen2_5_7b/llama3_1_8b/...)" )
    p2.add_argument("--custom_model", type=str, default=None)
    p2.add_argument("--z_dim", type=int, default=16)
    p2.add_argument("--c_ucb1", type=float, default=1.0)
    p2.add_argument("--alpha_linucb", type=float, default=1.0)
    p2.add_argument("--c_sim", type=float, default=1.0)
    p2.add_argument("--offline_freq", type=int, default=500)
    p2.add_argument("--offline_epochs", type=int, default=10)
    p2.add_argument("--warmup_epochs", type=int, default=15)
    p2.add_argument("--no_cache", action="store_true")
    p2.add_argument("--dump_prompts", type=int, default=3, help="Dump first N prompts to results/ourmethod_debug/<dataset>/")
    p2.add_argument("--ablation_mode", type=str, default="none", choices=["none", "s1_only", "s2_only", "s3_only"], help="Run ablation study by restricting to a single source")

    # plot
    p3 = sub.add_parser("plot", help="Plot regret curves for selected algorithms")
    p3.add_argument("--datasets", nargs="+", default=["newsgroups"], help=f"Supported: {sorted(DATASET_REGISTRY.keys())}")
    p3.add_argument("--algorithms", nargs="+", default=["all_baselines"],
                    help="Selectors: all_baselines | <baseline_name> | ourmethod:<model>")
    p3.add_argument("--n_rounds", type=int, default=2000, help="Original number of rounds the experiment was run for (to locate the folder)")
    p3.add_argument("--limit_rounds", type=int, default=None, help="If set, only plot and summarize the first N rounds from the loaded data.")
    p3.add_argument("--seed", type=int, default=42)

    # full pipeline
    p4 = sub.add_parser("full", help="Run baselines + OurMethod + regret plots for datasets")
    p4.add_argument("--datasets", nargs="+", default=["statlog", "magic", "covertype", "mnist", "adult", "census"],
                    help=f"Supported: {sorted(DATASET_REGISTRY.keys())}")
    p4.add_argument("--n_rounds", type=int, default=10000)
    p4.add_argument("--cold_start_n", type=int, default=100)
    p4.add_argument("--seed", type=int, default=42)
    p4.add_argument("--force", action="store_true", help="Rerun baselines even if cached logs exist")
    p4.add_argument("--model", dest="model_name", type=str, default="qwen2_5_7b",
                    help="LLM model alias (stub/smollm2/qwen2_5_7b/llama3_1_8b/...)")
    p4.add_argument("--custom_model", type=str, default=None)
    p4.add_argument("--z_dim", type=int, default=32)
    p4.add_argument("--c_ucb1", type=float, default=1.0)
    p4.add_argument("--alpha_linucb", type=float, default=1.0)
    p4.add_argument("--c_sim", type=float, default=1.0)
    p4.add_argument("--offline_freq", type=int, default=200)
    p4.add_argument("--offline_epochs", type=int, default=10)
    p4.add_argument("--warmup_epochs", type=int, default=15)
    p4.add_argument("--no_cache", action="store_true")
    p4.add_argument("--dump_prompts", type=int, default=3)
    p4.add_argument("--ablation_mode", type=str, default="none", choices=["none", "s1_only", "s2_only", "s3_only"])
    p4.add_argument("--use_llm_prior", action="store_true", help="Enable LLM prior for algorithms that require priors")
    p4.add_argument("--prior_source", type=str, default="local_llm", choices=["local_llm", "precomputed"], help="Prior provider backend")
    p4.add_argument("--precomputed_prior_path", type=str, default=None, help="Path to precomputed priors when prior_source=precomputed")
    p4.add_argument("--prior_prompt_style", type=str, default="auto", help="Prompt style hint used by prompt registry")
    p4.add_argument(
        "--llm_model",
        "--llm_model_path",
        dest="llm_model",
        type=str,
        default=None,
        help="LLM model alias or local model directory for AutoModelForCausalLM",
    )
    p4.add_argument("--llm_tokenizer_path", type=str, default=None, help="Local tokenizer directory (defaults to model path)")
    p4.add_argument("--llm_device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    p4.add_argument("--llm_dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="Inference dtype")
    p4.add_argument("--llm_cache_dir", type=str, default=os.path.join(RESULTS_DIR, "prior_cache"), help="Disk cache directory for priors")
    p4.add_argument("--rebuild_prior_cache", action="store_true", help="Rebuild prior cache before run")
    p4.add_argument("--disable_prior_cache", action="store_true", help="Disable cross-run prior cache reuse")
    p4.add_argument("--prior_dump_prompts", type=int, default=3, help="Dump first N prior prompts to results/prior_debug/<dataset>/")
    p4.add_argument("--prior_debug_dir", type=str, default=os.path.join(RESULTS_DIR, "prior_debug"), help="Debug output directory for dumped prior prompts")

    args = parser.parse_args()

    if args.command in ("baselines", "full") and getattr(args, "llm_model", None) and not getattr(args, "use_llm_prior", False):
        # Keep CLI behavior intuitive: providing a model implies user intends to use LLM prior.
        args.use_llm_prior = True

    if args.command == "baselines":
        cmd_baselines(args)
    elif args.command == "ourmethod":
        cmd_ourmethod(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "full":
        cmd_full(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

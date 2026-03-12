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
import os
import glob

import argparse
import json
import sys
sys.path.insert(0, os.path.abspath("."))
from OurMethod.extract_ucb import extract_ucb_stats
import re
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bandits.core.contextual_bandit import ContextualBandit
from bandits.data.data_sampler import (
    sample_newsgroups_data,
    sample_statlog_data,
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

RESULTS_DIR = os.path.join(ROOT, "results")


# ====================================================================== #
#  Registry                                                               #
# ====================================================================== #

DatasetLoader = Callable[[int, int, int], Tuple[ContextualBandit, np.ndarray, np.ndarray, int, int, int]]
BaselineFactory = Callable[[int, int], Any]

DATASET_REGISTRY: Dict[str, DatasetLoader] = {}
BASELINE_REGISTRY: Dict[str, BaselineFactory] = {}


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

    np.random.seed(seed)
    total = n_rounds + cold_start_n
    dataset, (opt_r_all, opt_a_all) = sample_newsgroups_data(data_path, total, shuffle_rows=True)

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
    return cmab, opt_rewards, opt_actions, num_actions, context_dim, n_rounds


def load_statlog(n_rounds: int, cold_start_n: int, seed: int):
    data_path = os.path.join(ROOT, "datasets", "statlog.trn")
    if not os.path.exists(data_path):
        print(f"[ERROR] {data_path} not found.")
        sys.exit(1)

    np.random.seed(seed)
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

    np.random.seed(seed)
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
):
    np.random.seed(seed)
    total = n_rounds + cold_start_n
    dataset, (opt_r_all, opt_a_all) = sampler(total, True)

    actual_total = len(opt_r_all)
    if total > actual_total:
        n_rounds = actual_total - cold_start_n
        print(f"[WARN] adjusted n_rounds={n_rounds}")

    num_actions = int(len(np.unique(opt_a_all)))
    context_dim = int(dataset.shape[1] - num_actions)

    opt_rewards = opt_r_all[cold_start_n: cold_start_n + n_rounds]
    opt_actions = opt_a_all[cold_start_n: cold_start_n + n_rounds]

    cmab = ContextualBandit(num_actions, context_dim)
    cmab.feed_data(dataset)
    return cmab, opt_rewards, opt_actions, num_actions, context_dim, n_rounds


def load_magic(n_rounds: int, cold_start_n: int, seed: int):
    return _load_generic_uci(sample_magic_data, n_rounds, cold_start_n, seed)


def load_mnist(n_rounds: int, cold_start_n: int, seed: int):
    return _load_generic_uci(sample_mnist_data, n_rounds, cold_start_n, seed)


def load_adult(n_rounds: int, cold_start_n: int, seed: int):
    return _load_generic_uci(sample_adult_data, n_rounds, cold_start_n, seed)


def load_census(n_rounds: int, cold_start_n: int, seed: int):
    return _load_generic_uci(sample_census_data, n_rounds, cold_start_n, seed)


def load_covertype(n_rounds: int, cold_start_n: int, seed: int):
    return _load_generic_uci(sample_covertype_data, n_rounds, cold_start_n, seed)

register_dataset("newsgroups", load_newsgroups)
register_dataset("statlog", load_statlog)
register_dataset("ag_news", load_ag_news)
register_dataset("magic", load_magic)
register_dataset("mnist", load_mnist)
register_dataset("adult", load_adult)
register_dataset("census", load_census)
register_dataset("cencus", load_census)
register_dataset("covertype", load_covertype)


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
        "training_freq": 100,
        "training_freq_network": 100,
        "training_epochs": 100,
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


def _b_neural_ucb(num_actions: int, context_dim: int):
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
        "lambda_prior": 1.0,
        "alpha": 1,
        "training_freq": 50,
        "training_epochs": 50,
    }
    return NeuralUCBSampling(h, name="neural_ucb")


def _b_neural_linucb(num_actions: int, context_dim: int):
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
        "lambda_prior": 1.0,
        "alpha": 0.5,
        "training_freq": 50,
        "training_epochs": 50,
    }
    return NeuralLinUCBSampling(h, name="neural_linucb")



register_baseline("ucb1", _b_ucb1)
register_baseline("linucb", _b_linucb)
register_baseline("epsilon_greedy", _b_epsilon_greedy)
register_baseline("neural_ucb", _b_neural_ucb)
register_baseline("neural_linucb", _b_neural_linucb)
register_baseline("neural_bandit", _b_neural_bandit)
register_baseline("neural_linear", _b_neural_linear)

BASELINE_NAMES = list(BASELINE_REGISTRY.keys())


# ====================================================================== #
#  IO helpers                                                             #
# ====================================================================== #

def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def _ourmethod_stem(model_name: str) -> str:
    return f"ourmethod__{_sanitize_name(model_name)}"


def _exp_dir(dataset_name: str, n_rounds: int, seed: int) -> str:
    d = os.path.join(RESULTS_DIR, f"{dataset_name}_{n_rounds}r_seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def save_algo_log(dataset_name: str, algo_stem: str, data: dict, n_rounds: int, seed: int, meta: Dict[str, Any] | None = None):
    exp_d = _exp_dir(dataset_name, n_rounds, seed)
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
    exp_d = _exp_dir(dataset_name, n_rounds, seed)
    
    # Support timestamp prefixes (e.g. 20260306_123456_my_algo.npz)
    pattern = os.path.join(exp_d, f"*_{algo_stem}.npz")
    matches = glob.glob(pattern)
    
    # Also check exact match
    fp_exact = os.path.join(exp_d, f"{algo_stem}.npz")
    if os.path.exists(fp_exact):
        matches.append(fp_exact)
        
    if not matches:
        return None
        
    # Get the latest edited file
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

def run_one_baseline(algo, algo_name, cmab, offset, n_rounds, opt_rewards, opt_actions):
    actions = np.zeros(n_rounds, dtype=np.int64)
    rewards = np.zeros(n_rounds)
    t0 = time.time()
    log_freq = max(1, n_rounds // 20)

    for i in range(n_rounds):
        ctx = cmab.context(offset + i)
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
    "ucb1": {"color": "#2ca02c", "ls": "--", "lw": 1.2},
    "linucb": {"color": "#d62728", "ls": "--", "lw": 1.2},
    "epsilon_greedy": {"color": "#9467bd", "ls": "--", "lw": 1.2},
    "neural_ucb": {"color": "#8c564b", "ls": "-.", "lw": 1.5},
    "neural_linucb": {"color": "#e377c2", "ls": "-.", "lw": 1.5},
    "neural_bandit": {"color": "#1f77b4", "ls": "-", "lw": 1.5},
    "neural_linear": {"color": "#ff7f0e", "ls": "-", "lw": 1.5},
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


def print_summary(all_data: Dict[str, dict], n_rounds: int):
    print(f"\n{'=' * 78}")
    print(f"{'Algorithm':<28s} {'Cum.Regret':>12s} {'Avg.Reward':>12s} {'Accuracy':>10s}")
    print(f"{'=' * 78}")
    for name, data in all_data.items():
        cr = data["cumulative_regret"][-1]
        avg_r = np.mean(data["rewards"])
        acc = np.mean(data["actions"] == data["opt_actions"])
        print(f"{name:<28s} {cr:>12.1f} {avg_r:>12.4f} {acc:>10.4f}")
    print(f"{'=' * 78}")

def plot_cumulative_reward(dataset_name: str, all_data: Dict[str, dict], n_rounds: int, seed: int, tag: str, save_rounds: int = None):
    ensure_plot_compatible(all_data)

    # Use the original experiment rounds (save_rounds) for folder saving if provided, otherwise fallback to n_rounds
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


def plot_average_reward(dataset_name: str, all_data: Dict[str, dict], n_rounds: int, seed: int, tag: str, save_rounds: int = None):

    ensure_plot_compatible(all_data)

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

def plot_regret(dataset_name: str, all_data: Dict[str, dict], n_rounds: int, seed: int, tag: str, save_rounds: int = None):
    ensure_plot_compatible(all_data)

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

        for bname in baselines:
            existing = None if args.force else load_algo_log(dname, bname, n_rounds, seed)
            if existing is not None:
                print(f"  [SKIP] {bname} (loaded existing)")
                all_data[bname] = existing
                continue

            print(f"  Running {bname} ...")
            t_start = time.time()
            np.random.seed(seed)
            algo = BASELINE_REGISTRY[bname](num_actions, context_dim)
            result = run_one_baseline(
                algo, bname, cmab, offset=cold_start_n,
                n_rounds=n_rounds, opt_rewards=opt_rewards, opt_actions=opt_actions,
            )
            t_total = time.time() - t_start
            result["times"] = {"total": t_total}
            
            meta = {
                "dataset": dname,
                "seed": seed,
                "n_rounds": n_rounds,
                "cold_start_n": cold_start_n,
                "algorithm": bname,
                "type": "baseline",
                "saved_at": datetime.now().isoformat(),
                **vars(args),
            }
            # 使用时间戳作为命名前缀
            record_stem = f"{run_timestamp}_{bname}"
            save_algo_log(dname, record_stem, result, n_rounds, seed, meta=meta)
            all_data[bname] = result

        if all_data:
            print_summary(all_data, n_rounds)
            tag = f"{run_timestamp}_baselines_" + _sanitize_name("_".join(baselines))
            plot_regret(dname, all_data, n_rounds, seed, tag=tag)

            summary = {
                "meta": {
                    "dataset": dname,
                    "n_rounds": n_rounds,
                    "seed": seed,
                    "cold_start_n": cold_start_n,
                    "baselines": baselines,
                    **vars(args),
                },
                "results": {
                    name: {
                        "cumulative_regret": float(data["cumulative_regret"][-1]),
                        "average_reward": float(np.mean(data["rewards"])),
                        "accuracy": float(np.mean(data["actions"] == data["opt_actions"])),
                        "times": data.get("times", {}),
                    }
                    for name, data in all_data.items()
                },
            }
            fp = os.path.join(_exp_dir(dname, n_rounds, seed), f"{run_timestamp}_baselines_summary.json")
            with open(fp, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"  Summary saved: {fp}")


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

        np.random.seed(seed)
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

        save_algo_log(dname, record_stem, result, n_rounds, seed, meta=meta)
        print(
            f"  OurMethod[{model_for_name}] cum_regret={result['cumulative_regret'][-1]:.1f}  "
            f"avg_reward={np.mean(result['rewards']):.4f}"
        )

        # Generate separate OurMethod Summary Report
        try:
            exp_d = _exp_dir(dname, n_rounds, seed)
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
        fp = os.path.join(_exp_dir(dname, n_rounds, seed), f"{record_stem}_summary.json")
        with open(fp, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  Summary saved: {fp}")


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
        
        # Pass actual_plot_rounds to limit the x-axis properly, but pass n_rounds for the directory lookup
        plot_regret(dname, loaded, actual_plot_rounds, seed, tag=tag, save_rounds=n_rounds)
        plot_cumulative_reward(dname, loaded, actual_plot_rounds, seed, tag=tag, save_rounds=n_rounds)
        plot_average_reward(dname, loaded, actual_plot_rounds, seed, tag=tag, save_rounds=n_rounds)

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
        fp = os.path.join(_exp_dir(dname, n_rounds, seed), f"{tag}_summary.json")
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
    p1.add_argument("--datasets", nargs="+", default=["newsgroups"], help=f"Supported: {sorted(DATASET_REGISTRY.keys())}")
    p1.add_argument("--baselines", nargs="+", default=["all_baselines"], help="Baseline selectors: all_baselines or subset names")
    p1.add_argument("--n_rounds", type=int, default=10000)
    p1.add_argument("--cold_start_n", type=int, default=100)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--force", action="store_true", help="Ignore existing logs and rerun")

    # ourmethod
    p2 = sub.add_parser("ourmethod", help="Run OurMethod on selected datasets")
    p2.add_argument("--datasets", nargs="+", default=["newsgroups"], help=f"Supported: {sorted(DATASET_REGISTRY.keys())}")
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

    args = parser.parse_args()

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

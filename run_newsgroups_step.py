#!/usr/bin/env python3
"""
分步实验脚本：基线 & OurMethod 分开跑，每个算法单独记录日志。

用法:
  # Step 1: 跑 7 个基线 + 绘制基线遗憾曲线
  python run_newsgroups_step.py baselines --n_rounds 2000 --seed 42

  # Step 2: 跑 OurMethod (smollm2)
  python run_newsgroups_step.py ourmethod --n_rounds 2000 --seed 42 --model smollm2

  # Step 3: 合并绘图
  python run_newsgroups_step.py plot --n_rounds 2000 --seed 42
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bandits.core.contextual_bandit import ContextualBandit
from bandits.data.data_sampler import sample_newsgroups_data

from bandits.algorithms.neural_bandit_model import NeuralBanditModel
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.algorithms.ucb1_sampling import UCB1Sampling
from bandits.algorithms.linucb_sampling import LinUCBSampling
from bandits.algorithms.epsilon_greedy_sampling import EpsilonGreedySampling
from bandits.algorithms.neural_ucb_sampling import NeuralUCBSampling
from bandits.algorithms.neural_linucb_sampling import NeuralLinUCBSampling

RESULTS_DIR = os.path.join(ROOT, "results")


def _exp_dir(n_rounds: int, seed: int) -> str:
    """Return the dedicated experiment directory, e.g. results/newsgroups_2000r_seed42/."""
    d = os.path.join(RESULTS_DIR, f"newsgroups_{n_rounds}r_seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d


# ====================================================================== #
#  Data loading                                                           #
# ====================================================================== #

def load_newsgroups(n_rounds: int, cold_start_n: int, seed: int):
    """Load dataset and return (cmab, opt_rewards, opt_actions, num_actions, context_dim, actual_n_rounds)."""
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
        print(f"[WARN] Adjusted n_rounds={n_rounds}")

    num_actions = 6
    context_dim = dataset.shape[1] - num_actions
    opt_rewards = opt_r_all[cold_start_n: cold_start_n + n_rounds]
    opt_actions = opt_a_all[cold_start_n: cold_start_n + n_rounds]

    cmab = ContextualBandit(num_actions, context_dim)
    cmab.feed_data(dataset)
    return cmab, opt_rewards, opt_actions, num_actions, context_dim, n_rounds


# ====================================================================== #
#  Algorithm creation                                                     #
# ====================================================================== #

def create_algorithm(name: str, num_actions: int, context_dim: int) -> Any:
    if name == "neural_bandit":
        h = {"context_dim": context_dim, "num_actions": num_actions,
             "layer_sizes": [100, 100], "activation": "relu",
             "initial_lr": 0.001, "batch_size": 512, "init_scale": 0.3,
             "use_dropout": False, "dropout_rate": 0.1, "layer_norm": False,
             "verbose": False}
        return NeuralBanditModel(h, name="neural_bandit")
    elif name == "neural_linear":
        h = {"context_dim": context_dim, "num_actions": num_actions,
             "layer_sizes": [100, 100], "activation": "relu",
             "initial_lr": 0.001, "batch_size": 512, "init_scale": 0.3,
             "use_dropout": False, "dropout_rate": 0.1, "layer_norm": False,
             "verbose": False, "lambda_prior": 0.25,
             "a0": 6, "b0": 6, "training_freq": 100,
             "training_freq_network": 100, "training_epochs": 100,
             "initial_pulls": 2}
        return NeuralLinearPosteriorSampling(h, name="neural_linear")
    elif name == "ucb1":
        h = argparse.Namespace(num_actions=num_actions, alpha=2.0)
        return UCB1Sampling("ucb1", h)
    elif name == "linucb":
        h = argparse.Namespace(num_actions=num_actions, context_dim=context_dim, alpha=1.0)
        return LinUCBSampling("linucb", h)
    elif name == "epsilon_greedy":
        h = argparse.Namespace(num_actions=num_actions, epsilon=0.1, epsilon_decay=0.0)
        return EpsilonGreedySampling("epsilon_greedy", h)
    elif name == "neural_ucb":
        h = {"context_dim": context_dim, "num_actions": num_actions,
             "layer_sizes": [100, 100], "activation": "relu",
             "initial_lr": 0.001, "batch_size": 512, "init_scale": 0.3,
             "use_dropout": False, "dropout_rate": 0.1, "layer_norm": False,
             "verbose": False, "lambda_prior": 1.0,
             "exploration_bonus": 0.1, "training_freq": 50,
             "training_epochs": 50}
        return NeuralUCBSampling(h, name="neural_ucb")
    elif name == "neural_linucb":
        h = {"context_dim": context_dim, "num_actions": num_actions,
             "layer_sizes": [100, 100], "activation": "relu",
             "initial_lr": 0.001, "batch_size": 512, "init_scale": 0.3,
             "use_dropout": False, "dropout_rate": 0.1, "layer_norm": False,
             "verbose": False, "lambda_prior": 1.0,
             "alpha": 0.5, "training_freq": 50, "training_epochs": 50}
        return NeuralLinUCBSampling(h, name="neural_linucb")
    raise ValueError(f"Unknown: {name}")


# ====================================================================== #
#  Run one algorithm                                                      #
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
            cr = np.sum(opt_rewards[:i+1] - rewards[:i+1])
            print(f"  [{algo_name}] step {i:>5d}/{n_rounds}  cum_regret={cr:.1f}")

    elapsed = time.time() - t0
    step_reg = opt_rewards[:n_rounds] - rewards
    cum_reg = np.cumsum(step_reg)
    cum_rew = np.cumsum(rewards)
    print(f"  [{algo_name}] done — cum_regret={cum_reg[-1]:.1f}  "
          f"avg_reward={np.mean(rewards):.4f}  time={elapsed:.1f}s")
    return {
        "actions": actions, "rewards": rewards,
        "opt_rewards": opt_rewards[:n_rounds],
        "opt_actions": opt_actions[:n_rounds],
        "step_regret": step_reg,
        "cumulative_regret": cum_reg,
        "cumulative_reward": cum_rew,
    }


# ====================================================================== #
#  Save / Load per-algorithm logs                                         #
# ====================================================================== #

def save_algo_log(algo_name: str, data: dict, n_rounds: int, seed: int):
    """Save one algorithm's step-by-step data as .npz into the experiment dir."""
    exp_d = _exp_dir(n_rounds, seed)
    fp = os.path.join(exp_d, f"{algo_name}.npz")
    np.savez_compressed(fp,
                        actions=data["actions"],
                        rewards=data["rewards"],
                        opt_rewards=data["opt_rewards"],
                        opt_actions=data["opt_actions"],
                        step_regret=data["step_regret"],
                        cumulative_regret=data["cumulative_regret"],
                        cumulative_reward=data["cumulative_reward"])
    print(f"  Saved: {fp}")
    return fp


def load_algo_log(algo_name: str, n_rounds: int, seed: int) -> dict | None:
    exp_d = _exp_dir(n_rounds, seed)
    fp = os.path.join(exp_d, f"{algo_name}.npz")
    if not os.path.exists(fp):
        return None
    d = np.load(fp)
    return {k: d[k] for k in d.files}


# ====================================================================== #
#  Plotting helpers                                                       #
# ====================================================================== #

STYLES = {
    "neural_bandit":    {"color": "#1f77b4", "ls": "-",  "lw": 1.5},
    "neural_linear":    {"color": "#ff7f0e", "ls": "-",  "lw": 1.5},
    "ucb1":             {"color": "#2ca02c", "ls": "--", "lw": 1.2},
    "linucb":           {"color": "#d62728", "ls": "--", "lw": 1.2},
    "epsilon_greedy":   {"color": "#9467bd", "ls": "--", "lw": 1.2},
    "neural_ucb":       {"color": "#8c564b", "ls": "-.", "lw": 1.5},
    "neural_linucb":    {"color": "#e377c2", "ls": "-.", "lw": 1.5},
    "combined_ucb":     {"color": "#17becf", "ls": "-",  "lw": 2.5},
}


def plot_regret(all_data: Dict[str, dict], n_rounds: int, seed: int,
                tag: str, title_suffix: str = ""):
    exp_d = _exp_dir(n_rounds, seed)
    steps = np.arange(1, n_rounds + 1)
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, data in all_data.items():
        s = STYLES.get(name, {"color": "gray", "ls": "-", "lw": 1})
        ax.plot(steps, data["cumulative_regret"], label=name, **s)
    ax.set_xlabel("Time Step", fontsize=13)
    ax.set_ylabel("Cumulative Regret", fontsize=13)
    ax.set_title(f"20 Newsgroups — Cumulative Regret ({n_rounds}r, seed={seed}){title_suffix}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(exp_d, f"{tag}_regret.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {fp}")
    return fp


def plot_reward(all_data: Dict[str, dict], n_rounds: int, seed: int,
                tag: str, title_suffix: str = ""):
    exp_d = _exp_dir(n_rounds, seed)
    steps = np.arange(1, n_rounds + 1)
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, data in all_data.items():
        s = STYLES.get(name, {"color": "gray", "ls": "-", "lw": 1})
        ax.plot(steps, data["cumulative_reward"], label=name, **s)
    opt_cum = np.cumsum(next(iter(all_data.values()))["opt_rewards"])
    ax.plot(steps, opt_cum, label="optimal", color="red", ls=":", lw=2)
    ax.set_xlabel("Time Step", fontsize=13)
    ax.set_ylabel("Cumulative Reward", fontsize=13)
    ax.set_title(f"20 Newsgroups — Cumulative Reward ({n_rounds}r, seed={seed}){title_suffix}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(exp_d, f"{tag}_reward.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {fp}")
    return fp


def print_summary(all_data, n_rounds):
    print(f"\n{'=' * 70}")
    print(f"{'Algorithm':<20s} {'Cum.Regret':>12s} {'Avg.Reward':>12s} {'Accuracy':>10s}")
    print(f"{'=' * 70}")
    for name, data in all_data.items():
        cr = data["cumulative_regret"][-1]
        avg_r = np.mean(data["rewards"])
        acc = np.mean(data["actions"] == data["opt_actions"])
        print(f"{name:<20s} {cr:>12.1f} {avg_r:>12.4f} {acc:>10.4f}")
    print(f"{'=' * 70}")


# ====================================================================== #
#  Sub-commands                                                           #
# ====================================================================== #

def cmd_baselines(args):
    """Run 7 baselines, save individual logs, plot baseline-only regret."""
    n_rounds = args.n_rounds
    seed = args.seed
    cold_start_n = args.cold_start_n

    cmab, opt_rewards, opt_actions, num_actions, context_dim, n_rounds = \
        load_newsgroups(n_rounds, cold_start_n, seed)

    print(f"=== Baselines on 20 Newsgroups ===")
    print(f"  n_rounds={n_rounds}, seed={seed}, cold_start_n={cold_start_n}")
    print(f"  context_dim={context_dim}, num_actions={num_actions}\n")

    baseline_names = [
        "neural_bandit", "neural_linear",
        "ucb1", "linucb", "epsilon_greedy",
        "neural_ucb", "neural_linucb",
    ]

    all_data = {}
    for bname in baseline_names:
        # Skip if log already exists
        existing = load_algo_log(bname, n_rounds, seed)
        if existing is not None:
            print(f"\n{'─' * 50}")
            print(f"[SKIP] {bname} — log already exists, loading ...")
            all_data[bname] = existing
            continue
        print(f"\n{'─' * 50}")
        print(f"Running {bname} ...")
        np.random.seed(seed)
        algo = create_algorithm(bname, num_actions, context_dim)
        result = run_one_baseline(
            algo, bname, cmab, offset=cold_start_n,
            n_rounds=n_rounds, opt_rewards=opt_rewards, opt_actions=opt_actions)
        all_data[bname] = result
        save_algo_log(bname, result, n_rounds, seed)

    print_summary(all_data, n_rounds)

    print("\nPlotting baselines-only curves ...")
    plot_regret(all_data, n_rounds, seed, tag="baselines", title_suffix=" [baselines]")
    plot_reward(all_data, n_rounds, seed, tag="baselines", title_suffix=" [baselines]")
    print("\nBaselines done!")


def cmd_ourmethod(args):
    """Run OurMethod (frozen LLM), save log."""
    from OurMethod.run_full_experiment import run_our_method_pipeline
    from OurMethod.core.protocol import Arm
    from OurMethod.core.prompt_builder import StructuredPromptBuilder

    n_rounds = args.n_rounds
    seed = args.seed
    cold_start_n = args.cold_start_n
    model_name = args.model_name

    cmab, opt_rewards, opt_actions, num_actions, context_dim, n_rounds = \
        load_newsgroups(n_rounds, cold_start_n, seed)

    print(f"=== OurMethod on 20 Newsgroups ===")
    print(f"  n_rounds={n_rounds}, seed={seed}, cold_start_n={cold_start_n}")
    print(f"  model={model_name}\n")

    # ---- Newsgroups-specific Arms & Prompt ---- #
    NEWSGROUP_CATEGORIES = [
        ("comp.graphics",           "Computer graphics discussions"),
        ("rec.sport.baseball",      "Baseball sports discussions"),
        ("sci.med",                 "Medical science discussions"),
        ("sci.space",               "Space science discussions"),
        ("talk.politics.guns",      "Gun politics discussions"),
        ("soc.religion.christian",  "Christian religion discussions"),
    ]
    newsgroup_arms = [
        Arm(arm_id=i, name=cat_name,
            description=cat_desc)
        for i, (cat_name, cat_desc) in enumerate(NEWSGROUP_CATEGORIES)
    ]
    newsgroup_pb = StructuredPromptBuilder(
        role="text topic classification expert",
        scenario=(
            "20 Newsgroups topic classification as contextual bandit. "
            "Given TF-IDF/LSA semantic features of a document, predict which "
            "newsgroup topic it belongs to. The features are 50-dimensional "
            "latent semantic vectors derived from the document text."
        ),
        max_feedback=5,
        max_feature_display=10,          # 50维只显示前10维，缩短prompt加速
    )

    np.random.seed(seed)
    result = run_our_method_pipeline(
        cmab=cmab,
        opt_rewards=opt_rewards,
        opt_actions=opt_actions,
        n_rounds=n_rounds,
        num_actions=num_actions,
        context_dim=context_dim,
        cold_start_n=cold_start_n,
        model_name=model_name,
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
        dataset_name="newsgroups",
        no_cache=args.no_cache,
        arms=newsgroup_arms,
        prompt_builder=newsgroup_pb,
    )

    if result:
        save_algo_log("combined_ucb", result, n_rounds, seed)
        print(f"\n  OurMethod cum_regret={result['cumulative_regret'][-1]:.1f}  "
              f"avg_reward={np.mean(result['rewards']):.4f}")
    else:
        print("[ERROR] OurMethod returned empty result.")
    print("\nOurMethod done!")


def cmd_plot(args):
    """Load all saved logs and plot combined figures."""
    n_rounds = args.n_rounds
    seed = args.seed

    algo_names = [
        "neural_bandit", "neural_linear",
        "ucb1", "linucb", "epsilon_greedy",
        "neural_ucb", "neural_linucb",
        "combined_ucb",
    ]

    all_data = {}
    for name in algo_names:
        d = load_algo_log(name, n_rounds, seed)
        if d is not None:
            all_data[name] = d
            print(f"  Loaded {name}")
        else:
            print(f"  [SKIP] {name} — log not found")

    if not all_data:
        print("[ERROR] No logs found.")
        return

    print_summary(all_data, n_rounds)

    print("\nPlotting combined curves ...")
    plot_regret(all_data, n_rounds, seed, tag="all", title_suffix="")
    plot_reward(all_data, n_rounds, seed, tag="all", title_suffix="")

    # Save JSON summary
    summary = {}
    for name, data in all_data.items():
        summary[name] = {
            "cumulative_regret": float(data["cumulative_regret"][-1]),
            "average_reward": float(np.mean(data["rewards"])),
            "accuracy": float(np.mean(data["actions"] == data["opt_actions"])),
        }
    summary["meta"] = {"dataset": "newsgroups", "n_rounds": n_rounds, "seed": seed}
    exp_d = _exp_dir(n_rounds, seed)
    fp = os.path.join(exp_d, f"summary.json")
    with open(fp, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {fp}")
    print("\nAll done!")


# ====================================================================== #
#  Main                                                                   #
# ====================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Newsgroups step-by-step experiment")
    sub = parser.add_subparsers(dest="command")

    # baselines
    p1 = sub.add_parser("baselines", help="Run 7 baseline algorithms")
    p1.add_argument("--n_rounds", type=int, default=2000)
    p1.add_argument("--cold_start_n", type=int, default=50)
    p1.add_argument("--seed", type=int, default=42)

    # ourmethod
    p2 = sub.add_parser("ourmethod", help="Run OurMethod with frozen LLM")
    p2.add_argument("--n_rounds", type=int, default=2000)
    p2.add_argument("--cold_start_n", type=int, default=50)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--model", dest="model_name", type=str, default="smollm2")
    p2.add_argument("--custom_model", type=str, default=None)
    p2.add_argument("--z_dim", type=int, default=16)
    p2.add_argument("--c_ucb1", type=float, default=1.0)
    p2.add_argument("--alpha_linucb", type=float, default=1.0)
    p2.add_argument("--c_sim", type=float, default=1.0)
    p2.add_argument("--offline_freq", type=int, default=500)
    p2.add_argument("--offline_epochs", type=int, default=10)
    p2.add_argument("--warmup_epochs", type=int, default=15)
    p2.add_argument("--no_cache", action="store_true")

    # plot
    p3 = sub.add_parser("plot", help="Combine all logs and plot")
    p3.add_argument("--n_rounds", type=int, default=2000)
    p3.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "baselines":
        cmd_baselines(args)
    elif args.command == "ourmethod":
        cmd_ourmethod(args)
    elif args.command == "plot":
        cmd_plot(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

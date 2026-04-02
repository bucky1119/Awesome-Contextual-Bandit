#!/usr/bin/env python3
"""MovieLens paper-style stage-1 runner with algorithm registry and run/plot commands."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from movielens_bandit_stage1.dataset import MovieLensBanditDataset
from movielens_bandit_stage1.linucb_shared_candidate import LinUCBSharedCandidate
from movielens_bandit_stage1.neural_linucb_shared_candidate import NeuralLinUCBSharedCandidate
from movielens_bandit_stage1.utils import ensure_dir, save_json, set_global_seed

RESULTS_DIR = os.path.join(ROOT, "results", "movielens_stage1")


AlgorithmFactory = Callable[[argparse.Namespace, MovieLensBanditDataset], Any]
ALGO_REGISTRY: Dict[str, AlgorithmFactory] = {}


def register_algo(name: str, factory: AlgorithmFactory) -> None:
    ALGO_REGISTRY[name] = factory


def _algo_linucb_shared(args: argparse.Namespace, dataset: MovieLensBanditDataset):
    return LinUCBSharedCandidate(
        arm_dim=dataset.arm_dim,
        context_dim=dataset.context_dim,
        num_arms=dataset.num_arms,
        alpha=args.alpha,
        lambda_prior=args.lambda_prior,
        fit_intercept=args.fit_intercept,
        initial_pulls=args.initial_pulls,
    )


def _algo_neural_linucb_shared(args: argparse.Namespace, dataset: MovieLensBanditDataset):
    return NeuralLinUCBSharedCandidate(
        arm_dim=dataset.arm_dim,
        context_dim=dataset.context_dim,
        num_arms=dataset.num_arms,
        latent_dim=args.latent_dim,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        alpha=args.alpha,
        lambda_prior=args.lambda_prior,
        fit_intercept=args.fit_intercept,
        initial_pulls=args.initial_pulls,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_every=args.train_every,
        buffer_size=args.buffer_size,
        dropout=args.dropout,
        seed=args.seed,
    )


register_algo("linucb_shared", _algo_linucb_shared)
register_algo("neural_linucb_shared", _algo_neural_linucb_shared)


STYLES = {
    "linucb_shared": {"color": "#d62728", "ls": "--", "lw": 1.8},
    "neural_linucb_shared": {"color": "#1f77b4", "ls": "-", "lw": 1.8},
}


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    out = np.zeros_like(arr, dtype=np.float64)
    cumsum = np.cumsum(np.insert(arr.astype(np.float64), 0, 0.0))
    for i in range(len(arr)):
        left = max(0, i + 1 - window)
        total = cumsum[i + 1] - cumsum[left]
        out[i] = total / float(i + 1 - left)
    return out


def _build_exp_dir(seed: int, n_rounds: int, algos: List[str], timestamp: str | None = None) -> str:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    algo_tag = _sanitize_name("_".join(algos))
    folder = f"{ts}_movielens_paper_env_{n_rounds}r_seed{seed}_{algo_tag}"
    exp_dir = os.path.join(RESULTS_DIR, folder)
    ensure_dir(exp_dir)
    return exp_dir


def _find_latest_exp_dir(seed: int, n_rounds: int) -> str | None:
    pattern = os.path.join(RESULTS_DIR, f"*_movielens_paper_env_{n_rounds}r_seed{seed}_*")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return sorted(matches, key=os.path.getmtime)[-1]


def _run_one_algo(
    algo_name: str,
    algo: Any,
    dataset: MovieLensBanditDataset,
    n_rounds: int,
    log_every: int,
    reward_seed: int,
) -> Dict[str, Any]:
    actions = np.zeros(n_rounds, dtype=np.int64)
    rewards = np.zeros(n_rounds, dtype=np.float32)
    regrets = np.zeros(n_rounds, dtype=np.float32)
    hits = np.zeros(n_rounds, dtype=np.float32)
    optimal_rewards = np.zeros(n_rounds, dtype=np.float32)
    chosen_mus = np.zeros(n_rounds, dtype=np.float32)
    optimal_mus = np.zeros(n_rounds, dtype=np.float32)

    rng = np.random.default_rng(reward_seed)
    t0 = time.time()

    for t in range(n_rounds):
        # rd = dataset.get_round(t)
        real_t = t % len(dataset)
        rd = dataset.get_round(real_t)

        arm_features = rd["arm_features"]   # [N, d_arm]
        context = rd["context"]             # [d_ctx]
        mu = rd["mu"]                       # [N]

        action = int(algo.action(arm_features, context))
        #xxx
        
        # reward = float(dataset.sample_reward(t, action, rng=rng))
        # regret = float(dataset.regret(t, action))
        
        reward = float(dataset.sample_reward(real_t, action, rng=rng))
        regret = float(dataset.regret(real_t, action))

        chosen_mu = float(mu[action])
        optimal_mu = float(np.max(mu))

        algo.update(arm_features, context, action, reward)

        actions[t] = action
        rewards[t] = reward
        regrets[t] = regret
        hits[t] = 1.0 if reward > 0.0 else 0.0
        optimal_rewards[t] = optimal_mu  # 保留命名兼容，但这里其实是 optimal expected reward
        chosen_mus[t] = chosen_mu
        optimal_mus[t] = optimal_mu

        if (t + 1) % max(1, log_every) == 0 or t == 0 or t + 1 == n_rounds:
            print(
                f"  [{algo_name}] step {t + 1:>6d}/{n_rounds} "
                f"cum_regret={np.sum(regrets[:t+1]):.4f} "
                f"avg_reward={np.mean(rewards[:t+1]):.4f} "
                f"avg_mu={np.mean(chosen_mus[:t+1]):.4f}"
            )

    elapsed = float(time.time() - t0)
    cumulative_regret = np.cumsum(regrets)
    cumulative_reward = np.cumsum(rewards)
    cumulative_accuracy = np.cumsum(hits) / np.arange(1, n_rounds + 1)
    cumulative_avg_mu = np.cumsum(chosen_mus) / np.arange(1, n_rounds + 1)

    return {
        "algorithm": algo_name,
        "actions": actions,
        "rewards": rewards,
        "regrets": regrets,
        "hits": hits,
        "optimal_rewards": optimal_rewards,
        "chosen_mus": chosen_mus,
        "optimal_mus": optimal_mus,
        "cumulative_regret": cumulative_regret,
        "cumulative_reward": cumulative_reward,
        "cumulative_accuracy": cumulative_accuracy,
        "cumulative_avg_mu": cumulative_avg_mu,
        "n_rounds": n_rounds,
        "run_time_seconds": elapsed,
        "final_metrics": {
            "cumulative_regret": float(cumulative_regret[-1]),
            "average_reward": float(np.mean(rewards)),
            "accuracy": float(np.mean(hits)),
            "average_mu": float(np.mean(chosen_mus)),
            "average_optimal_mu": float(np.mean(optimal_mus)),
            "regret_at_100": float(np.sum(regrets[: min(100, n_rounds)])),
            "regret_at_500": float(np.sum(regrets[: min(500, n_rounds)])),
            "run_time_seconds": elapsed,
        },
    }


def _save_algo_npz(exp_dir: str, result: Dict[str, Any]) -> str:
    fp = os.path.join(exp_dir, f"{result['algorithm']}.npz")
    np.savez_compressed(
        fp,
        actions=result["actions"],
        rewards=result["rewards"],
        regrets=result["regrets"],
        hits=result["hits"],
        optimal_rewards=result["optimal_rewards"],
        chosen_mus=result["chosen_mus"],
        optimal_mus=result["optimal_mus"],
        cumulative_regret=result["cumulative_regret"],
        cumulative_reward=result["cumulative_reward"],
        cumulative_accuracy=result["cumulative_accuracy"],
        cumulative_avg_mu=result["cumulative_avg_mu"],
        final_metrics_json=np.array(json.dumps(result["final_metrics"], ensure_ascii=False)),
    )
    return fp


def _load_algo_npz(fp: str) -> Dict[str, Any]:
    d = np.load(fp, allow_pickle=False)
    out = {
        "algorithm": os.path.splitext(os.path.basename(fp))[0],
        "actions": d["actions"],
        "rewards": d["rewards"],
        "regrets": d["regrets"],
        "hits": d["hits"],
        "cumulative_regret": d["cumulative_regret"],
        "cumulative_reward": d["cumulative_reward"],
        "cumulative_accuracy": d["cumulative_accuracy"],
        "final_metrics": json.loads(str(d["final_metrics_json"].item())),
    }
    if "optimal_rewards" in d.files:
        out["optimal_rewards"] = d["optimal_rewards"]
    if "chosen_mus" in d.files:
        out["chosen_mus"] = d["chosen_mus"]
    if "optimal_mus" in d.files:
        out["optimal_mus"] = d["optimal_mus"]
    if "cumulative_avg_mu" in d.files:
        out["cumulative_avg_mu"] = d["cumulative_avg_mu"]
    return out


def _plot_group_curves(exp_dir: str, run_name: str, results: Dict[str, Dict[str, Any]], rolling_window: int) -> Dict[str, str]:
    algo_names = list(results.keys())
    if not algo_names:
        return {}

    n_rounds = len(results[algo_names[0]]["rewards"])
    steps = np.arange(1, n_rounds + 1)

    for name in algo_names[1:]:
        if len(results[name]["rewards"]) != n_rounds:
            raise ValueError(f"Length mismatch among algorithms: {algo_names[0]} vs {name}")

    outputs: Dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(12, 7))
    for name in algo_names:
        style = STYLES.get(name, {"color": "gray", "ls": "-", "lw": 1.6})
        ax.plot(steps, results[name]["cumulative_regret"], label=name, **style)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Pseudo-Regret")
    ax.set_title(f"{run_name} - Cumulative Pseudo-Regret")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fp = os.path.join(exp_dir, "cumulative_regret_curve.png")
    fig.savefig(fp, dpi=260, bbox_inches="tight")
    plt.close(fig)
    outputs["cumulative_regret_curve"] = fp

    fig, ax = plt.subplots(figsize=(12, 7))
    for name in algo_names:
        style = STYLES.get(name, {"color": "gray", "ls": "-", "lw": 1.6})
        avg_reward = results[name]["cumulative_reward"] / steps
        ax.plot(steps, avg_reward, label=name, **style)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Average Sampled Reward")
    ax.set_title(f"{run_name} - Average Sampled Reward")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fp = os.path.join(exp_dir, "average_reward_curve.png")
    fig.savefig(fp, dpi=260, bbox_inches="tight")
    plt.close(fig)
    outputs["average_reward_curve"] = fp

    if all("chosen_mus" in results[name] for name in algo_names):
        fig, ax = plt.subplots(figsize=(12, 7))
        for name in algo_names:
            style = STYLES.get(name, {"color": "gray", "ls": "-", "lw": 1.6})
            avg_mu = np.cumsum(results[name]["chosen_mus"]) / steps
            ax.plot(steps, avg_mu, label=name, **style)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Average Expected Reward (mu)")
        ax.set_title(f"{run_name} - Average Expected Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fp = os.path.join(exp_dir, "average_mu_curve.png")
        fig.savefig(fp, dpi=260, bbox_inches="tight")
        plt.close(fig)
        outputs["average_mu_curve"] = fp

    fig, ax = plt.subplots(figsize=(12, 7))
    for name in algo_names:
        style = STYLES.get(name, {"color": "gray", "ls": "-", "lw": 1.6})
        rolling_acc = _rolling_mean(results[name]["hits"], rolling_window)
        ax.plot(steps, rolling_acc, label=name, **style)
    ax.set_xlabel("Time Step")
    ax.set_ylabel(f"Rolling Sampled Reward Rate (window={rolling_window})")
    ax.set_title(f"{run_name} - Sampled Reward Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fp = os.path.join(exp_dir, "accuracy_curve.png")
    fig.savefig(fp, dpi=260, bbox_inches="tight")
    plt.close(fig)
    outputs["accuracy_curve"] = fp

    return outputs


def _save_group_csv(exp_dir: str, results: Dict[str, Dict[str, Any]]) -> str:
    fp = os.path.join(exp_dir, "metrics.csv")
    names = list(results.keys())
    n_rounds = len(results[names[0]]["rewards"]) if names else 0

    with open(fp, "w", encoding="utf-8") as f:
        header = ["round"]
        for name in names:
            header.extend(
                [
                    f"{name}.reward",
                    f"{name}.regret",
                    f"{name}.hit",
                    f"{name}.cum_regret",
                    f"{name}.avg_reward",
                    f"{name}.accuracy",
                    f"{name}.avg_mu",
                ]
            )
        f.write(",".join(header) + "\n")

        for i in range(n_rounds):
            row = [str(i + 1)]
            for name in names:
                r = results[name]
                avg_r = float(r["cumulative_reward"][i]) / float(i + 1)
                acc = float(np.sum(r["hits"][: i + 1])) / float(i + 1)
                avg_mu = float(np.sum(r["chosen_mus"][: i + 1])) / float(i + 1) if "chosen_mus" in r else 0.0
                row.extend(
                    [
                        f"{float(r['rewards'][i]):.6f}",
                        f"{float(r['regrets'][i]):.6f}",
                        f"{float(r['hits'][i]):.6f}",
                        f"{float(r['cumulative_regret'][i]):.6f}",
                        f"{avg_r:.6f}",
                        f"{acc:.6f}",
                        f"{avg_mu:.6f}",
                    ]
                )
            f.write(",".join(row) + "\n")
    return fp


def _build_summary(run_meta: Dict[str, Any], results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "meta": run_meta,
        "results": {},
    }
    for name, r in results.items():
        summary["results"][name] = r["final_metrics"]
    return summary


def _save_summary_md(exp_dir: str, summary: Dict[str, Any]) -> str:
    fp = os.path.join(exp_dir, "run_summary.md")
    meta = summary["meta"]
    lines: List[str] = []
    lines.append("# MovieLens Paper-Style Run Summary")
    lines.append("")
    lines.append(f"- dataset: {meta['dataset']}")
    lines.append(f"- data_path: {meta['data_path']}")
    lines.append(f"- n_rounds: {meta['n_rounds']}")
    lines.append(f"- seed: {meta['seed']}")
    lines.append(f"- num_arms: {meta['num_arms']}")
    lines.append(f"- arm_dim: {meta['arm_dim']}")
    lines.append(f"- context_dim: {meta['context_dim']}")
    lines.append(f"- algorithms: {', '.join(meta['algorithms'])}")
    lines.append("")
    lines.append("## Final Metrics")
    lines.append("")
    lines.append("| Algorithm | Cum.Regret | Avg.Reward | Accuracy | Avg.Mu | Avg.Opt.Mu | Regret@100 | Regret@500 | Time(s) |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, m in summary["results"].items():
        lines.append(
            f"| {name} | {m['cumulative_regret']:.4f} | {m['average_reward']:.4f} | {m['accuracy']:.4f} | "
            f"{m['average_mu']:.4f} | {m['average_optimal_mu']:.4f} | "
            f"{m['regret_at_100']:.4f} | {m['regret_at_500']:.4f} | {m['run_time_seconds']:.2f} |"
        )
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append("- run.log")
    lines.append("- run_config.json")
    lines.append("- summary.json")
    lines.append("- run_summary.md")
    lines.append("- metrics.csv")
    lines.append("- cumulative_regret_curve.png")
    lines.append("- average_reward_curve.png")
    lines.append("- average_mu_curve.png")
    lines.append("- accuracy_curve.png")
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return fp


def _get_run_log_path(exp_dir: str) -> str:
    return os.path.join(exp_dir, "run.log")


def _log_line(exp_dir: str, msg: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line)
    with open(_get_run_log_path(exp_dir), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def cmd_run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    dataset = MovieLensBanditDataset.from_npz(args.data_path)
    dataset.validate()

    # n_rounds = min(int(args.n_rounds), len(dataset))
    #xxx
    n_rounds = int(args.n_rounds)
    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive and dataset must be non-empty")

    exp_dir = _build_exp_dir(seed=args.seed, n_rounds=n_rounds, algos=args.algorithms)
    ensure_dir(exp_dir)

    run_meta = {
        "dataset": "movielens_paper_env",
        "data_path": args.data_path,
        "n_rounds": int(n_rounds),
        "seed": int(args.seed),
        "num_arms": int(dataset.num_arms),
        "arm_dim": int(dataset.arm_dim),
        "context_dim": int(dataset.context_dim),
        "algorithms": args.algorithms,
        "created_at": datetime.now().isoformat(),
        "params": {
            "alpha": args.alpha,
            "lambda_prior": args.lambda_prior,
            "initial_pulls": args.initial_pulls,
            "fit_intercept": args.fit_intercept,
            "latent_dim": args.latent_dim,
            "hidden_size": args.hidden_size,
            "hidden_layers": args.hidden_layers,
            "train_every": args.train_every,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "dropout": args.dropout,
        },
    }
    save_json(run_meta, os.path.join(exp_dir, "run_config.json"))

    _log_line(exp_dir, "Run start")
    _log_line(exp_dir, f"exp_dir={exp_dir}")
    _log_line(
        exp_dir,
        f"dataset_size={len(dataset)} num_arms={dataset.num_arms} arm_dim={dataset.arm_dim} context_dim={dataset.context_dim}",
    )
    _log_line(exp_dir, f"algorithms={args.algorithms}")

    results: Dict[str, Dict[str, Any]] = {}
    for algo_name in args.algorithms:
        if algo_name not in ALGO_REGISTRY:
            raise ValueError(f"Unsupported algorithm: {algo_name}")

        set_global_seed(args.seed)
        _log_line(exp_dir, f"Running algorithm={algo_name}")
        algo = ALGO_REGISTRY[algo_name](args, dataset)
        result = _run_one_algo(
            algo_name,
            algo,
            dataset,
            n_rounds=n_rounds,
            log_every=args.log_every,
            reward_seed=args.seed,
        )
        results[algo_name] = result
        npz_fp = _save_algo_npz(exp_dir, result)
        _log_line(
            exp_dir,
            (
                f"Done algorithm={algo_name} "
                f"cum_regret={result['final_metrics']['cumulative_regret']:.4f} "
                f"avg_reward={result['final_metrics']['average_reward']:.4f} "
                f"avg_mu={result['final_metrics']['average_mu']:.4f} "
                f"saved_npz={npz_fp}"
            ),
        )

    csv_fp = _save_group_csv(exp_dir, results)
    plot_paths = _plot_group_curves(exp_dir, "movielens_paper_env", results, rolling_window=args.rolling_window)

    summary = _build_summary(run_meta, results)
    summary_json_fp = os.path.join(exp_dir, "summary.json")
    save_json(summary, summary_json_fp)
    summary_md_fp = _save_summary_md(exp_dir, summary)

    _log_line(exp_dir, f"Saved metrics_csv={csv_fp}")
    _log_line(exp_dir, f"Saved summary_json={summary_json_fp}")
    _log_line(exp_dir, f"Saved summary_md={summary_md_fp}")
    for k, v in plot_paths.items():
        _log_line(exp_dir, f"Saved plot {k}={v}")
    _log_line(exp_dir, "Run finished")

    print("=== MovieLens Paper-Style Result ===")
    print(f"saved_dir: {exp_dir}")
    for name, r in results.items():
        m = r["final_metrics"]
        print(
            f"[{name}] cum_regret={m['cumulative_regret']:.4f} "
            f"avg_reward={m['average_reward']:.4f} "
            f"avg_mu={m['average_mu']:.4f}"
        )


def _resolve_plot_dir(args: argparse.Namespace) -> str:
    if args.exp_dir:
        exp_dir = args.exp_dir
    else:
        exp_dir = _find_latest_exp_dir(seed=args.seed, n_rounds=args.n_rounds)
    if exp_dir is None or not os.path.isdir(exp_dir):
        raise FileNotFoundError("Cannot resolve experiment directory. Use --exp_dir or provide valid --n_rounds/--seed.")
    return exp_dir


def cmd_plot(args: argparse.Namespace) -> None:
    exp_dir = _resolve_plot_dir(args)

    npz_files = sorted(glob.glob(os.path.join(exp_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No algorithm npz logs found in {exp_dir}")

    selected = set(args.algorithms) if args.algorithms else None
    loaded: Dict[str, Dict[str, Any]] = {}

    for fp in npz_files:
        name = os.path.splitext(os.path.basename(fp))[0]
        if selected is not None and name not in selected:
            continue
        loaded[name] = _load_algo_npz(fp)

    if not loaded:
        raise ValueError("No algorithm logs loaded for plotting (check --algorithms).")

    plot_paths = _plot_group_curves(exp_dir, "movielens_paper_env_plot", loaded, rolling_window=args.rolling_window)

    summary = {
        "meta": {
            "exp_dir": exp_dir,
            "algorithms": list(loaded.keys()),
            "created_at": datetime.now().isoformat(),
        },
        "results": {name: data["final_metrics"] for name, data in loaded.items()},
        "plot_paths": plot_paths,
    }

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join(exp_dir, f"{tag}_plot_summary.json")
    save_json(summary, fp)
    print(f"Plot refreshed in: {exp_dir}")
    print(f"Plot summary saved: {fp}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MovieLens paper-style stage-1 experiment runner")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run one experiment with multiple algorithms")
    p_run.add_argument(
        "--algorithms",
        "--algo",
        dest="algorithms",
        nargs="+",
        default=["linucb_shared", "neural_linucb_shared"],
        choices=sorted(ALGO_REGISTRY.keys()),
        help="Algorithms to run together in one experiment directory",
    )
    p_run.add_argument(
        "--data_path",
        type=str,
        default="datasets/movielens_paper_env_200arms_seed42.npz",
    )
    p_run.add_argument("--n_rounds", type=int, default=5000)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--log_every", type=int, default=200)
    p_run.add_argument("--rolling_window", type=int, default=100)

    p_run.add_argument("--alpha", type=float, default=1.0)
    p_run.add_argument("--lambda_prior", type=float, default=1.0)
    p_run.add_argument("--initial_pulls", type=int, default=1)
    p_run.add_argument("--fit_intercept", dest="fit_intercept", action="store_true")
    p_run.add_argument("--no_fit_intercept", dest="fit_intercept", action="store_false")
    p_run.set_defaults(fit_intercept=True)

    p_run.add_argument("--latent_dim", type=int, default=32)
    p_run.add_argument("--hidden_size", type=int, default=64)
    p_run.add_argument("--hidden_layers", type=int, default=2)
    p_run.add_argument("--train_every", type=int, default=100)
    p_run.add_argument("--epochs", type=int, default=5)
    p_run.add_argument("--lr", type=float, default=1e-3)
    p_run.add_argument("--batch_size", type=int, default=64)
    p_run.add_argument("--buffer_size", type=int, default=10000)
    p_run.add_argument("--dropout", type=float, default=0.0)

    p_plot = sub.add_parser("plot", help="Re-plot curves from an existing experiment directory")
    p_plot.add_argument("--exp_dir", type=str, default=None, help="Explicit experiment directory under results/movielens_stage1")
    p_plot.add_argument("--n_rounds", type=int, default=5000, help="Used with --seed to find latest run when --exp_dir omitted")
    p_plot.add_argument("--seed", type=int, default=42, help="Used with --n_rounds to find latest run when --exp_dir omitted")
    p_plot.add_argument("--algorithms", nargs="+", default=None, help="Optional subset of algorithm names to plot")
    p_plot.add_argument("--rolling_window", type=int, default=100)

    return parser


def main() -> None:
    parser = _build_parser()

    if len(sys.argv) > 1 and sys.argv[1] not in {"run", "plot", "-h", "--help"}:
        argv = ["run", *sys.argv[1:]]
    else:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "plot":
        cmd_plot(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
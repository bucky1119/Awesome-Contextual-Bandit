#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验日志自动记录模块
======================================================
用途：
1. 为 bandit / LLM-bandit 实验自动生成结构化日志目录
2. 自动保存：
   - config.json
   - metrics.json
   - step_metrics.csv
   - curves.npz
   - 实验日志模板 markdown
3. 自动写入可复现所需的基础信息
4. 预留人工填写区域（关键观察 / 问题 / 下一步等）

设计原则：
- 不依赖项目内部复杂对象
- 只接收 run_dataset_step.py 已有的 result/meta/args 等字典
- 尽量保持健壮，避免日志记录影响主实验
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


# ============================================================
# 基础工具
# ============================================================

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _sanitize_name(name: str) -> str:
    """将名称清理成适合作为文件夹/文件名的形式。"""
    import re
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name))


def _json_safe(obj: Any) -> Any:
    """将常见不可 JSON 序列化对象转为安全格式。"""
    import argparse

    if isinstance(obj, argparse.Namespace):
        return {k: _json_safe(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return obj


def _write_json(fp: str, data: Dict[str, Any]) -> None:
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, ensure_ascii=False, indent=2)


def _get_git_commit() -> str:
    """尽量获取当前 git commit hash。失败则返回 unknown。"""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return out
    except Exception:
        return "unknown"


def _collect_runtime_info() -> Dict[str, Any]:
    """收集运行环境信息。"""
    info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": _get_git_commit(),
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        else:
            info["cuda_device_name"] = None
    except Exception:
        info["torch_version"] = "not_installed"
        info["cuda_available"] = False
        info["cuda_device_count"] = 0
        info["cuda_device_name"] = None

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except Exception:
        info["transformers_version"] = "not_installed"

    try:
        import matplotlib
        info["matplotlib_version"] = matplotlib.__version__
    except Exception:
        info["matplotlib_version"] = "not_installed"

    return info


def _safe_mean(x) -> Optional[float]:
    if x is None:
        return None
    x = np.asarray(x)
    if x.size == 0:
        return None
    return float(np.mean(x))


# ============================================================
# 指标提取
# ============================================================

def _build_auto_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 run_dataset_step.py 的 result 字典中抽取自动指标。
    约定 result 中通常已有：
      - rewards
      - cumulative_reward
      - step_regret
      - cumulative_regret
      - actions
      - opt_actions
      - times
      - hparams
    """
    rewards = np.asarray(result.get("rewards", []), dtype=float)
    cumulative_reward = np.asarray(result.get("cumulative_reward", []), dtype=float)
    step_regret = np.asarray(result.get("step_regret", []), dtype=float)
    cumulative_regret = np.asarray(result.get("cumulative_regret", []), dtype=float)
    actions = np.asarray(result.get("actions", []))
    opt_actions = np.asarray(result.get("opt_actions", []))

    metrics = {
        "num_rounds": int(len(rewards)),
        "final_cumulative_regret": float(cumulative_regret[-1]) if len(cumulative_regret) > 0 else None,
        "final_cumulative_reward": float(cumulative_reward[-1]) if len(cumulative_reward) > 0 else None,
        "average_reward": _safe_mean(rewards),
        "accuracy": float(np.mean(actions == opt_actions)) if len(actions) > 0 and len(actions) == len(opt_actions) else None,
    }

    # 早期 regret / reward
    for k in [100, 500, 1000]:
        if len(cumulative_regret) >= k:
            metrics[f"regret_at_{k}"] = float(cumulative_regret[k - 1])
        else:
            metrics[f"regret_at_{k}"] = None

        if len(rewards) >= k:
            metrics[f"avg_reward_first_{k}"] = float(np.mean(rewards[:k]))
        else:
            metrics[f"avg_reward_first_{k}"] = None

    # 后期 reward
    if len(rewards) >= 1000:
        metrics["avg_reward_last_1000"] = float(np.mean(rewards[-1000:]))
    else:
        metrics["avg_reward_last_1000"] = None

    # 时间统计
    times = result.get("times", {})
    if isinstance(times, dict):
        metrics["times"] = _json_safe(times)
    else:
        metrics["times"] = {}

    # 如果存在 trust 信息，也自动记录
    if "trust_weights" in result:
        trust_weights = np.asarray(result["trust_weights"], dtype=float)
        metrics["avg_trust_weight"] = _safe_mean(trust_weights)
        metrics["final_trust_weight"] = float(trust_weights[-1]) if len(trust_weights) > 0 else None

    if "prior_errors" in result:
        prior_errors = np.asarray(result["prior_errors"], dtype=float)
        metrics["avg_prior_error"] = _safe_mean(prior_errors)
        metrics["final_prior_error"] = float(prior_errors[-1]) if len(prior_errors) > 0 else None

    return metrics


# ============================================================
# 明细文件保存
# ============================================================

def _save_step_metrics_csv(log_dir: str, result: Dict[str, Any]) -> str:
    """
    保存逐轮指标 csv。
    """
    fp = os.path.join(log_dir, "step_metrics.csv")

    rewards = np.asarray(result.get("rewards", []))
    cumulative_reward = np.asarray(result.get("cumulative_reward", []))
    step_regret = np.asarray(result.get("step_regret", []))
    cumulative_regret = np.asarray(result.get("cumulative_regret", []))
    actions = np.asarray(result.get("actions", []))
    opt_actions = np.asarray(result.get("opt_actions", []))
    opt_rewards = np.asarray(result.get("opt_rewards", []))

    n = len(rewards)

    with open(fp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "reward",
            "cumulative_reward",
            "step_regret",
            "cumulative_regret",
            "action",
            "opt_action",
            "opt_reward",
        ])

        for i in range(n):
            writer.writerow([
                i + 1,
                float(rewards[i]) if i < len(rewards) else "",
                float(cumulative_reward[i]) if i < len(cumulative_reward) else "",
                float(step_regret[i]) if i < len(step_regret) else "",
                float(cumulative_regret[i]) if i < len(cumulative_regret) else "",
                int(actions[i]) if i < len(actions) else "",
                int(opt_actions[i]) if i < len(opt_actions) else "",
                float(opt_rewards[i]) if i < len(opt_rewards) else "",
            ])

    return fp


def _save_curves_npz(log_dir: str, result: Dict[str, Any]) -> str:
    """
    保存 numpy 曲线文件，便于后处理。
    """
    fp = os.path.join(log_dir, "curves.npz")
    payload = {}

    for key in [
        "actions", "rewards", "opt_rewards", "opt_actions",
        "step_regret", "cumulative_regret", "cumulative_reward"
    ]:
        if key in result:
            payload[key] = result[key]

    if "trust_weights" in result:
        payload["trust_weights"] = result["trust_weights"]
    if "prior_errors" in result:
        payload["prior_errors"] = result["prior_errors"]

    np.savez_compressed(fp, **payload)
    return fp


# ============================================================
# Markdown 模板
# ============================================================

def _build_markdown_template(
    *,
    dataset_name: str,
    algorithm_name: str,
    meta: Dict[str, Any],
    auto_metrics: Dict[str, Any],
    runtime_info: Dict[str, Any],
    prompt_info: Optional[Dict[str, Any]] = None,
    plot_paths: Optional[Dict[str, str]] = None,
) -> str:
    """
    构造实验日志 markdown 模板。
    自动填基础信息，保留人工填写区域。
    """
    prompt_info = prompt_info or {}
    plot_paths = plot_paths or {}

    lines = []
    lines.append(f"# 实验日志：{algorithm_name} on {dataset_name}")
    lines.append("")
    lines.append("## 1. 实验基本信息")
    lines.append("")
    lines.append(f"- **数据集**: {dataset_name}")
    lines.append(f"- **算法**: {algorithm_name}")
    lines.append(f"- **记录时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **随机种子**: {meta.get('seed', 'N/A')}")
    lines.append(f"- **实验轮数**: {meta.get('n_rounds', 'N/A')}")
    lines.append(f"- **冷启动轮数**: {meta.get('cold_start_n', 'N/A')}")
    lines.append(f"- **Git Commit**: {runtime_info.get('git_commit', 'unknown')}")
    lines.append("")

    lines.append("## 2. 运行环境")
    lines.append("")
    lines.append(f"- **Python**: {runtime_info.get('python', 'unknown')}")
    lines.append(f"- **Platform**: {runtime_info.get('platform', 'unknown')}")
    lines.append(f"- **Torch**: {runtime_info.get('torch_version', 'unknown')}")
    lines.append(f"- **CUDA 可用**: {runtime_info.get('cuda_available', 'unknown')}")
    lines.append(f"- **GPU**: {runtime_info.get('cuda_device_name', 'N/A')}")
    lines.append(f"- **Transformers**: {runtime_info.get('transformers_version', 'unknown')}")
    lines.append("")

    lines.append("## 3. 自动记录的核心指标")
    lines.append("")
    lines.append(f"- **最终累计遗憾**: {auto_metrics.get('final_cumulative_regret', 'N/A')}")
    lines.append(f"- **最终累计奖励**: {auto_metrics.get('final_cumulative_reward', 'N/A')}")
    lines.append(f"- **平均奖励**: {auto_metrics.get('average_reward', 'N/A')}")
    lines.append(f"- **准确率**: {auto_metrics.get('accuracy', 'N/A')}")
    lines.append(f"- **Regret@100**: {auto_metrics.get('regret_at_100', 'N/A')}")
    lines.append(f"- **Regret@500**: {auto_metrics.get('regret_at_500', 'N/A')}")
    lines.append(f"- **Regret@1000**: {auto_metrics.get('regret_at_1000', 'N/A')}")
    lines.append(f"- **前100轮平均奖励**: {auto_metrics.get('avg_reward_first_100', 'N/A')}")
    lines.append(f"- **前500轮平均奖励**: {auto_metrics.get('avg_reward_first_500', 'N/A')}")
    lines.append(f"- **前1000轮平均奖励**: {auto_metrics.get('avg_reward_first_1000', 'N/A')}")
    lines.append(f"- **后1000轮平均奖励**: {auto_metrics.get('avg_reward_last_1000', 'N/A')}")
    if "avg_trust_weight" in auto_metrics:
        lines.append(f"- **平均 Trust Weight**: {auto_metrics.get('avg_trust_weight', 'N/A')}")
    if "avg_prior_error" in auto_metrics:
        lines.append(f"- **平均 Prior Error**: {auto_metrics.get('avg_prior_error', 'N/A')}")
    lines.append("")

    if auto_metrics.get("times"):
        lines.append("## 4. 自动记录的时间统计")
        lines.append("")
        for k, v in auto_metrics["times"].items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("## 5. Prompt / Prior 信息")
    lines.append("")
    lines.append(f"- **Prior Backend**: {meta.get('prior_backend', 'N/A')}")
    lines.append(f"- **Prompt Template**: {meta.get('prompt_template', 'N/A')}")
    lines.append(f"- **Prompt Style**: {meta.get('prior_prompt_style', 'N/A')}")
    for k, v in prompt_info.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## 6. 自动生成文件")
    lines.append("")
    lines.append("- `config.json`：本次实验配置")
    lines.append("- `metrics.json`：自动提取的总体指标")
    lines.append("- `step_metrics.csv`：逐轮指标")
    lines.append("- `curves.npz`：曲线数组文件")
    if plot_paths:
        for k, v in plot_paths.items():
            lines.append(f"- **{k}**: {v}")
    lines.append("")

    # 以下留给你手写
    lines.append("## 7. 实验目标（请手写补充）")
    lines.append("")
    lines.append("- [ ] 本实验要验证什么？")
    lines.append("- [ ] 主要对照 baseline 是谁？")
    lines.append("- [ ] 预期现象是什么？")
    lines.append("")

    lines.append("## 8. 关键观察（请手写补充）")
    lines.append("")
    lines.append("1. ")
    lines.append("2. ")
    lines.append("3. ")
    lines.append("")

    lines.append("## 9. 问题 / Debug 记录（请手写补充）")
    lines.append("")
    lines.append("- ")
    lines.append("")

    lines.append("## 10. 下一步计划（请手写补充）")
    lines.append("")
    lines.append("- ")
    lines.append("")

    return "\n".join(lines)


# ============================================================
# 对外主接口
# ============================================================

def save_single_run_log(
    *,
    exp_dir: str,
    dataset_name: str,
    algorithm_name: str,
    args: Dict[str, Any],
    meta: Dict[str, Any],
    result: Dict[str, Any],
    prompt_info: Optional[Dict[str, Any]] = None,
    plot_paths: Optional[Dict[str, str]] = None,
) -> str:
    """
    保存单个算法单次实验的自动日志。

    参数说明：
    - exp_dir: 当前实验结果目录（通常就是 _exp_dir(...) 返回的目录）
    - dataset_name: 数据集名
    - algorithm_name: 算法名
    - args/meta/result: 来自 run_dataset_step.py 的现成对象
    - prompt_info: 可选，附加 prompt 信息
    - plot_paths: 可选，绘图路径

    返回：
    - 生成的日志目录路径
    """
    safe_algo = _sanitize_name(algorithm_name)
    log_dir = _ensure_dir(os.path.join(exp_dir, "experiment_logs", safe_algo))

    runtime_info = _collect_runtime_info()
    auto_metrics = _build_auto_metrics(result)

    # 1) 保存 config
    config = {
        "dataset_name": dataset_name,
        "algorithm_name": algorithm_name,
        "args": _json_safe(args),
        "meta": _json_safe(meta),
        "runtime_info": runtime_info,
        "created_at": datetime.now().isoformat(),
    }
    _write_json(os.path.join(log_dir, "config.json"), config)

    # 2) 保存 metrics
    _write_json(os.path.join(log_dir, "metrics.json"), auto_metrics)

    # 3) 保存逐轮 csv
    _save_step_metrics_csv(log_dir, result)

    # 4) 保存曲线 npz
    _save_curves_npz(log_dir, result)

    # 5) 保存 markdown 模板
    md = _build_markdown_template(
        dataset_name=dataset_name,
        algorithm_name=algorithm_name,
        meta=meta,
        auto_metrics=auto_metrics,
        runtime_info=runtime_info,
        prompt_info=prompt_info,
        plot_paths=plot_paths,
    )
    with open(os.path.join(log_dir, "log_template.md"), "w", encoding="utf-8") as f:
        f.write(md)

    return log_dir


def save_dataset_summary_log(
    *,
    exp_dir: str,
    dataset_name: str,
    summary_name: str,
    args: Dict[str, Any],
    summary_meta: Dict[str, Any],
    summary_results: Dict[str, Any],
    plot_paths: Optional[Dict[str, str]] = None,
) -> str:
    """
    保存一个数据集级别的 summary 日志。
    适合 baselines 全部跑完之后统一记录。
    """
    safe_name = _sanitize_name(summary_name)
    log_dir = _ensure_dir(os.path.join(exp_dir, "experiment_logs", safe_name))

    runtime_info = _collect_runtime_info()

    # 保存 summary config
    config = {
        "dataset_name": dataset_name,
        "summary_name": summary_name,
        "args": _json_safe(args),
        "summary_meta": _json_safe(summary_meta),
        "runtime_info": runtime_info,
        "created_at": datetime.now().isoformat(),
    }
    _write_json(os.path.join(log_dir, "config.json"), config)

    _write_json(os.path.join(log_dir, "summary_results.json"), summary_results)

    lines = []
    lines.append(f"# 数据集级实验总结：{summary_name}")
    lines.append("")
    lines.append(f"- **数据集**: {dataset_name}")
    lines.append(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Git Commit**: {runtime_info.get('git_commit', 'unknown')}")
    lines.append("")

    lines.append("## 1. 自动记录的结果汇总")
    lines.append("")
    for algo_name, algo_res in summary_results.items():
        lines.append(f"### {algo_name}")
        for k, v in algo_res.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    if plot_paths:
        lines.append("## 2. 图像文件")
        lines.append("")
        for k, v in plot_paths.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("## 3. 关键观察（请手写补充）")
    lines.append("")
    lines.append("1. ")
    lines.append("2. ")
    lines.append("3. ")
    lines.append("")

    lines.append("## 4. 下一步（请手写补充）")
    lines.append("")
    lines.append("- ")
    lines.append("")

    with open(os.path.join(log_dir, "summary_log_template.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return log_dir
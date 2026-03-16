"""
下载并预处理 Covertype、MNIST、Statlog、Adult 四个数据集，保存为 bandit 格式 .npz 文件。
同时扫描 datasets/ 目录下所有数据集，生成统一说明文档。

用法:
    python prepare_datasets.py

输出:
    datasets/covertype.npz
    datasets/mnist.npz
    datasets/statlog.npz
    datasets/adult.npz
    datasets/README.md   (datasets/ 目录下所有数据集的说明文档)
"""

from __future__ import annotations

# ── macOS SSL 证书修复（必须在所有网络相关 import 之前）──────────────────────
import ssl
import certifi
import urllib.request

# 全局替换默认 HTTPS 上下文，使用 certifi 证书束
_ssl_ctx = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: _ssl_ctx

import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────

def classification_to_bandit_problem(contexts: np.ndarray, labels: np.ndarray):
    """Convert (X, y) classification dataset → bandit .npz format.

    Returns
    -------
    dataset : ndarray, shape (n, context_dim + num_actions)
        Columns: [context features | one-hot reward matrix]
    opt_rewards : ndarray, shape (n,)  — always 1.0
    opt_actions : ndarray, shape (n,)  — the correct class index
    """
    unique_labels, normalized_labels = np.unique(labels, return_inverse=True)
    num_actions = len(unique_labels)
    n = len(contexts)
    rewards = np.zeros((n, num_actions), dtype=np.float32)
    rewards[np.arange(n), normalized_labels] = 1.0
    dataset = np.hstack([contexts.astype(np.float32), rewards])
    return dataset, np.ones(n, dtype=np.float32), normalized_labels.astype(np.int32)


def save_npz(path: str, dataset, opt_rewards, opt_actions, meta: dict):
    """Save bandit dataset to .npz with metadata.

    Always stores ``num_actions`` and ``context_dim`` so ``check_npz`` can read
    them back reliably without heuristics.
    """
    n, total_cols = dataset.shape
    # Infer from opt_actions range if not explicitly supplied
    if "num_actions" not in meta:
        meta = dict(meta)
        meta["num_actions"] = int(total_cols - (dataset.shape[1] - len(np.unique(opt_actions))))
    if "context_dim" not in meta:
        meta = dict(meta)
        meta["context_dim"] = int(total_cols - meta["num_actions"])
    np.savez_compressed(
        path,
        dataset=dataset,
        opt_rewards=opt_rewards,
        opt_actions=opt_actions,
        **{f"meta_{k}": np.array(v) for k, v in meta.items()},
    )


def check_npz(path: str) -> dict:
    """Load .npz and return shape / integrity info.

    Uses stored ``meta_num_actions`` / ``meta_context_dim`` when available
    (written by ``save_npz``), so the result is always reliable regardless of
    whether context features are binary.
    """
    data = np.load(path, allow_pickle=True)
    ds = data["dataset"]
    n, total_cols = ds.shape
    opt_r = data["opt_rewards"]
    opt_a = data["opt_actions"]

    # ── resolve num_actions ───────────────────────────────────────────────
    if "meta_num_actions" in data:
        num_actions = int(data["meta_num_actions"])
    else:
        # Fallback: use opt_actions range
        num_actions = int(opt_a.max()) + 1
    context_dim = total_cols - num_actions

    ok_dataset = not np.any(np.isnan(ds))
    ok_opt_r   = len(opt_r) == n and not np.any(np.isnan(opt_r))
    ok_opt_a   = len(opt_a) == n
    ok_rewards = np.all(ds[:, context_dim:].sum(axis=1) == 1.0)

    return {
        "n_samples": n,
        "context_dim": context_dim,
        "num_actions": num_actions,
        "file_size_mb": os.path.getsize(path) / 1e6,
        "ok_no_nan": ok_dataset,
        "ok_opt_r": ok_opt_r,
        "ok_opt_a": ok_opt_a,
        "ok_rewards": ok_rewards,
        "healthy": ok_dataset and ok_opt_r and ok_opt_a and ok_rewards,
    }


# ────────────────────────────────────────────────────────────────────────────
#  1. Covertype
# ────────────────────────────────────────────────────────────────────────────

def prepare_covertype():
    out = os.path.join(DATASETS_DIR, "covertype.npz")
    print("\n" + "=" * 60)
    print("  Covertype  (sklearn.datasets.fetch_covtype)")
    print("=" * 60)

    t0 = time.time()
    from sklearn.datasets import fetch_covtype
    print("  Fetching from sklearn (cache: ~/scikit_learn_data) …")
    ds = fetch_covtype()
    X = ds.data.astype(np.float32)   # (581012, 54)
    y = ds.target.astype(int)         # 1-indexed labels 1..7

    print(f"  Raw shape : {X.shape},  labels : {sorted(np.unique(y).tolist())}")
    print(f"  Num classes : {len(np.unique(y))} arms")

    dataset, opt_rewards, opt_actions = classification_to_bandit_problem(X, y)
    num_actions = len(np.unique(y))
    save_npz(out, dataset, opt_rewards, opt_actions, {
        "source": "sklearn.datasets.fetch_covtype / UCI id=31",
        "name": "Forest Covertype",
        "note": "54 cartographic features; 7 forest cover types",
        "num_actions": num_actions,
        "context_dim": X.shape[1],
    })
    elapsed = time.time() - t0
    print(f"  Saved  → {out}  ({os.path.getsize(out)/1e6:.1f} MB, {elapsed:.1f}s)")
    return out


# ────────────────────────────────────────────────────────────────────────────
#  2. MNIST
# ────────────────────────────────────────────────────────────────────────────

def prepare_mnist():
    out = os.path.join(DATASETS_DIR, "mnist.npz")
    print("\n" + "=" * 60)
    print("  MNIST (OpenML mnist_784)")
    print("=" * 60)

    t0 = time.time()
    from sklearn.datasets import fetch_openml
    print("  Fetching from OpenML (may take a while the first time) …")
    X_raw, y_raw = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")

    X = X_raw.astype(np.float32)
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    y = y_raw.astype(int)

    print(f"  Raw shape : {X.shape},  labels range : {y.min()} – {y.max()}")
    print(f"  Classes   : {sorted(np.unique(y).tolist())}  → {len(np.unique(y))} arms")

    dataset, opt_rewards, opt_actions = classification_to_bandit_problem(X, y)
    num_actions = len(np.unique(y))
    save_npz(out, dataset, opt_rewards, opt_actions, {
        "source": "OpenML mnist_784 version=1",
        "name": "MNIST",
        "note": "pixel values normalized to [0,1]",
        "num_actions": num_actions,
        "context_dim": X.shape[1],
    })
    elapsed = time.time() - t0
    print(f"  Saved  → {out}  ({os.path.getsize(out)/1e6:.1f} MB, {elapsed:.1f}s)")
    return out


# ────────────────────────────────────────────────────────────────────────────
#  3. Statlog (Shuttle)
# ────────────────────────────────────────────────────────────────────────────

def prepare_statlog():
    out = os.path.join(DATASETS_DIR, "statlog.npz")
    src = os.path.join(DATASETS_DIR, "statlog.trn")
    print("\n" + "=" * 60)
    print("  Statlog (Shuttle) — 从 statlog.trn 转换")
    print("=" * 60)

    if not os.path.exists(src):
        raise FileNotFoundError(f"原始文件不存在: {src}")

    t0 = time.time()
    data = np.loadtxt(src)                     # shape (43500, 10)
    X = data[:, :-1].astype(np.float32)        # 9 数值特征
    y = data[:, -1].astype(int)                # 标签 1–7

    print(f"  Raw shape : {X.shape},  labels range : {y.min()} – {y.max()}")
    classes = sorted(np.unique(y).tolist())
    print(f"  Classes   : {classes}  → {len(classes)} arms")

    # 统一映射为 0-indexed（1→0, 2→1, …, 7→6）
    label_map = {v: i for i, v in enumerate(classes)}
    y0 = np.array([label_map[v] for v in y], dtype=int)

    dataset, opt_rewards, opt_actions = classification_to_bandit_problem(X, y0)
    num_actions = len(classes)
    save_npz(out, dataset, opt_rewards, opt_actions, {
        "source": "UCI ML Repository id=148",
        "name": "Statlog Shuttle",
        "note": "labels mapped 1-7 → 0-6",
        "num_actions": num_actions,
        "context_dim": X.shape[1],
    })
    elapsed = time.time() - t0
    print(f"  Saved  → {out}  ({os.path.getsize(out)/1e6:.1f} MB, {elapsed:.1f}s)")
    return out


# ────────────────────────────────────────────────────────────────────────────
#  4. Adult Income (Census Income)
# ────────────────────────────────────────────────────────────────────────────

_ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
_ADULT_CAT_COLS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]


def prepare_adult():
    out = os.path.join(DATASETS_DIR, "adult.npz")
    src = os.path.join(DATASETS_DIR, "adult.data")
    print("\n" + "=" * 60)
    print("  Adult Income — 从 adult.data 转换")
    print("=" * 60)

    if not os.path.exists(src):
        raise FileNotFoundError(f"原始文件不存在: {src}")

    t0 = time.time()
    df = pd.read_csv(src, header=None, names=_ADULT_COLUMNS,
                     na_values=" ?", skipinitialspace=True)
    n_before = len(df)
    df = df.dropna()
    print(f"  Rows: {n_before} raw → {len(df)} after dropping NaN")

    # one-hot 编码分类列
    df = pd.get_dummies(df, columns=_ADULT_CAT_COLS, drop_first=False, dtype=np.float32)

    # 标签列：规范化后映射为 0/1
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    df["income"] = (df["income"].str.contains(">50K")).astype(int)

    feature_cols = [c for c in df.columns if c != "income"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["income"].values.astype(int)

    print(f"  Feature dim : {X.shape[1]}  (数值 + one-hot 后)")
    print(f"  Classes     : {sorted(np.unique(y).tolist())}  → 2 arms (0=≤50K, 1=>50K)")

    dataset, opt_rewards, opt_actions = classification_to_bandit_problem(X, y)
    save_npz(out, dataset, opt_rewards, opt_actions, {
        "source": "UCI ML Repository id=2",
        "name": "Adult Income",
        "note": "categorical cols one-hot encoded; income mapped to 0/1",
        "num_actions": 2,
        "context_dim": X.shape[1],
    })
    elapsed = time.time() - t0
    print(f"  Saved  → {out}  ({os.path.getsize(out)/1e6:.1f} MB, {elapsed:.1f}s)")
    return out


# ────────────────────────────────────────────────────────────────────────────
#  Integrity check & summary
# ────────────────────────────────────────────────────────────────────────────

# 所有数据集的元信息（包括新下载的和已有的）
# 键名 = datasets/ 目录下的文件名
ALL_DATASETS_META = {
    # ── 本次脚本下载 ───────────────────────────────────────────────────────
    "covertype.npz": {
        "full_name": "Forest Covertype",
        "source": "sklearn.datasets.fetch_covtype / UCI id=31",
        "url": "https://archive.ics.uci.edu/dataset/31/covertype",
        "task": "7分类",
        "format": "npz",
        "prepare_script": "prepare_datasets.py",
        "description": "预测科罗拉多州荒野地区的森林覆盖类型，特征包括海拔、坡度、土壤类型（one-hot）等 54 个制图变量。",
    },
    "mnist.npz": {
        "full_name": "MNIST Handwritten Digits",
        "source": "OpenML mnist_784 version=1",
        "url": "https://www.openml.org/d/554",
        "task": "10分类（数字 0–9）",
        "format": "npz",
        "prepare_script": "prepare_datasets.py",
        "description": "手写数字灰度图像，28×28 像素展平为 784 维向量，像素值归一化至 [0,1]。",
    },
    # ── 已有数据集 ─────────────────────────────────────────────────────────
    "statlog.npz": {
        "full_name": "Statlog (Shuttle)",
        "source": "UCI ML Repository id=148",
        "url": "https://archive.ics.uci.edu/dataset/148/statlog+shuttle",
        "task": "7分类",
        "format": "npz",
        "prepare_script": "prepare_datasets.py",
        "description": "NASA 航天飞机传感器数据，9 个数值特征，用于推断飞行控制系统的工作状态（7 类）。标签由 1–7 映射为 0–6。",
    },
    "newsgroups.npz": {
        "full_name": "20 Newsgroups（6 类，LSA 50 维）",
        "source": "sklearn.datasets.fetch_20newsgroups",
        "url": "http://qwone.com/~jason/20Newsgroups/",
        "task": "6分类",
        "format": "npz",
        "prepare_script": "prepare_newsgroups.py",
        "description": "新闻文章文本，经 TF-IDF 向量化后用 TruncatedSVD 降维至 50 维语义特征；选取 6 个主题差异较大的类别。",
    },
    "magic.npz": {
        "full_name": "MAGIC Gamma Telescope",
        "source": "UCI ML Repository id=159",
        "url": "https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope",
        "task": "2分类",
        "format": "npz",
        "prepare_script": "bandits/data/data_sampler.py (sample_magic_data)",
        "description": "大气切伦科夫望远镜模拟数据，10 个数值特征，区分 gamma 射线事件与背景强子噪声。",
    },
    "ag_news.npz": {
        "full_name": "AG News（TF-IDF + SVD 100 维）",
        "source": "HuggingFace datasets ag_news",
        "url": "https://huggingface.co/datasets/fancyzhx/ag_news",
        "task": "4分类（World / Sports / Business / Sci&Tech）",
        "format": "npz",
        "prepare_script": "（本地脚本生成）",
        "description": "新闻标题+摘要，TF-IDF 向量化后用 SVD 降维至 100 维语义特征。",
    },
    "ag_news_tfidf_svd100.npz": {
        "full_name": "AG News TF-IDF SVD100（备用版本）",
        "source": "HuggingFace datasets ag_news",
        "url": "https://huggingface.co/datasets/fancyzhx/ag_news",
        "task": "4分类",
        "format": "npz",
        "prepare_script": "（本地脚本生成）",
        "description": "同 ag_news.npz，另一处理流程生成的备用版本。",
    },
    "adult.npz": {
        "full_name": "Adult Income（Census Income）",
        "source": "UCI ML Repository id=2",
        "url": "https://archive.ics.uci.edu/dataset/2/adult",
        "task": "2分类（收入 >50K / ≤50K）",
        "format": "npz",
        "prepare_script": "prepare_datasets.py",
        "description": "1994 年美国人口普查数据，8 个分类特征经 one-hot 编码后共约 108 维，预测年收入是否超过 $50K（0=≤50K, 1=>50K）。",
    },
}


def print_integrity_table(infos: dict[str, dict]):
    print("\n" + "=" * 80)
    print("  .npz 数据完整性检查结果")
    print("=" * 80)
    header = f"{'文件':<30} {'样本数':>9} {'特征维度':>8} {'臂数':>6} {'文件大小':>9} {'状态'}"
    print(header)
    print("-" * 80)
    for fname, info in infos.items():
        status = "✅ OK" if info["healthy"] else "❌ FAIL"
        issues = []
        if not info["ok_no_nan"]:   issues.append("NaN")
        if not info["ok_opt_r"]:    issues.append("opt_r")
        if not info["ok_opt_a"]:    issues.append("opt_a")
        if not info["ok_rewards"]:  issues.append("rewards")
        if issues:
            status += "  [" + ", ".join(issues) + "]"
        print(f"{fname:<30} {info['n_samples']:>9,} {info['context_dim']:>8} "
              f"{info['num_actions']:>6} {info['file_size_mb']:>8.1f}MB  {status}")
    print("=" * 80)


# ────────────────────────────────────────────────────────────────────────────
#  README.md generator
# ────────────────────────────────────────────────────────────────────────────

def generate_readme(npz_infos: dict[str, dict]):
    """生成 datasets/README.md，覆盖目录下所有数据集文件。

    npz_infos: fname → check_npz() 返回的 dict（仅 .npz 文件有）
    """
    today = date.today().strftime("%Y-%m-%d")

    # 扫描 datasets/ 目录，找出实际存在的文件
    existing_files = set(os.listdir(DATASETS_DIR))
    # 按预定顺序排列已知文件，末尾追加未知文件
    ordered = [f for f in ALL_DATASETS_META if f in existing_files]
    unknown = sorted(f for f in existing_files
                     if f not in ALL_DATASETS_META
                     and not f.startswith(".") and f != "README.md")
    all_files = ordered + unknown

    lines = [
        "# Datasets 说明文档",
        "",
        f"> 自动生成于 {today} | 脚本: `prepare_datasets.py`",
        "",
        "本目录存放所有用于上下文赌博机（Contextual Bandit）实验的数据集。",
        "",
        "`.npz` 文件统一采用 **Bandit 格式**：",
        "",
        "```",
        "foo.npz",
        "├── dataset     — float32 (n_samples, context_dim + num_actions)",
        "│                  前 context_dim 列为上下文特征，后 num_actions 列为 one-hot 奖励矩阵",
        "├── opt_rewards — float32 (n_samples,)   最优奖励（全为 1.0）",
        "└── opt_actions — int32   (n_samples,)   最优动作索引（即真实类别标签）",
        "```",
        "",
        "---",
        "",
        "## 数据集总览",
        "",
        "| 文件 | 全名 | 样本数 | 特征维度 | 臂数 | 文件大小 | 任务 | 格式 |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |",
    ]

    for fname in all_files:
        meta = ALL_DATASETS_META.get(fname, {})
        full = meta.get("full_name", fname)
        task = meta.get("task", "—")
        fmt  = meta.get("format", "npz" if fname.endswith(".npz") else "raw")
        fpath = os.path.join(DATASETS_DIR, fname)
        size_str = f"{os.path.getsize(fpath)/1e6:.1f} MB" if os.path.exists(fpath) else "—"

        if fname in npz_infos:
            info = npz_infos[fname]
            n    = f"{info['n_samples']:,}"
            cdim = str(info["context_dim"])
            arms = str(info["num_actions"])
        else:
            n, cdim, arms = "—", "—", "—"

        lines.append(
            f"| `{fname}` | {full} | {n} | {cdim} | {arms} | {size_str} | {task} | {fmt} |"
        )

    lines += ["", "---", "", "## 各数据集详情", ""]

    for fname in all_files:
        meta  = ALL_DATASETS_META.get(fname, {})
        info  = npz_infos.get(fname, {})
        full  = meta.get("full_name", fname)
        fpath = os.path.join(DATASETS_DIR, fname)
        size_str = f"{os.path.getsize(fpath)/1e6:.1f} MB" if os.path.exists(fpath) else "—"

        lines += [f"### {full}", ""]
        lines += [f"- **文件**: `{fname}`"]
        if meta.get("source"):  lines.append(f"- **来源**: {meta['source']}")
        if meta.get("url"):     lines.append(f"- **URL**: {meta['url']}")
        if meta.get("task"):    lines.append(f"- **任务**: {meta['task']}")
        if meta.get("format"):  lines.append(f"- **格式**: {meta['format']}")
        if meta.get("prepare_script"): lines.append(f"- **生成脚本**: `{meta['prepare_script']}`")

        # 实际统计（仅 .npz）
        if info:
            lines += [
                f"- **样本数 (n_samples)**: {info['n_samples']:,}",
                f"- **特征维度 (context_dim)**: {info['context_dim']}",
                f"- **臂数 (num_actions)**: {info['num_actions']}",
                f"- **文件大小**: {size_str}",
                f"- **完整性**: {'✅ 通过' if info.get('healthy') else '❌ 异常'}",
            ]
        else:
            lines.append(f"- **文件大小**: {size_str}")

        if meta.get("description"): lines.append(f"- **描述**: {meta['description']}")
        lines.append("")

    lines += [
        "---",
        "",
        "## 加载方式",
        "",
        "```python",
        "import numpy as np",
        "",
        "data = np.load('datasets/covertype.npz')",
        "dataset     = data['dataset']       # (n, context_dim + num_actions)",
        "opt_rewards = data['opt_rewards']   # (n,)",
        "opt_actions = data['opt_actions']   # (n,)",
        "",
        "num_actions = <已知臂数>            # 例如 covertype=7, mnist=10",
        "context_dim = dataset.shape[1] - num_actions",
        "contexts    = dataset[:, :context_dim]",
        "rewards     = dataset[:, context_dim:]",
        "```",
        "",
        "或使用 `bandits/data/data_sampler.py` 中封装的函数按需采样。",
        "",
    ]

    readme_path = os.path.join(DATASETS_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  README 已生成 → {readme_path}  ({len(all_files)} 个数据集）")
    return readme_path


# ────────────────────────────────────────────────────────────────────────────
#  main
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 60)
    print("  下载并预处理数据集")
    print("█" * 60)

    errors = []

    # 1. Covertype
    try:
        prepare_covertype()
    except Exception as e:
        print(f"  [ERROR] Covertype: {e}")
        errors.append(("covertype", str(e)))

    # 2. MNIST
    try:
        prepare_mnist()
    except Exception as e:
        print(f"  [ERROR] MNIST: {e}")
        errors.append(("mnist", str(e)))

    # 3. Statlog
    try:
        prepare_statlog()
    except Exception as e:
        print(f"  [ERROR] Statlog: {e}")
        errors.append(("statlog", str(e)))

    # 4. Adult Income
    try:
        prepare_adult()
    except Exception as e:
        print(f"  [ERROR] Adult: {e}")
        errors.append(("adult", str(e)))

    # ── Integrity check：扫描 datasets/ 下所有 .npz 文件 ─────────────────
    print("\n" + "=" * 60)
    print("  完整性检查（datasets/ 下全部 .npz 文件）")
    print("=" * 60)

    npz_infos = {}
    for fname in sorted(os.listdir(DATASETS_DIR)):
        if not fname.endswith(".npz"):
            continue
        fpath = os.path.join(DATASETS_DIR, fname)
        try:
            npz_infos[fname] = check_npz(fpath)
            ok = "✅" if npz_infos[fname]["healthy"] else "❌"
            info = npz_infos[fname]
            print(f"  {ok} {fname:<30}  "
                  f"n={info['n_samples']:,}  "
                  f"ctx={info['context_dim']}  "
                  f"arms={info['num_actions']}  "
                  f"{info['file_size_mb']:.1f}MB")
        except Exception as e:
            print(f"  ❌ {fname}: {e}")

    print_integrity_table(npz_infos)

    # ── Generate README (covers ALL files in datasets/) ──────────────────
    generate_readme(npz_infos)

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  完成")
    if errors:
        print("  以下数据集下载/处理失败:")
        for name, msg in errors:
            print(f"    - {name}: {msg}")
    else:
        print("  Covertype、MNIST、Statlog、Adult 均已成功处理并验证")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()

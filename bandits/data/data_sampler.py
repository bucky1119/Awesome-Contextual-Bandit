# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions to create bandit problems from datasets (PyTorch/NumPy version)."""

# ============================================================================
# 文件说明（详细版：输入 / 输出 / 功能）
#
# 一、这个文件解决什么问题
# - 将多种来源的数据（UCI、本地 .npz、文本向量化结果等）统一转换为“上下文 Bandit”可直接使用的标准格式。
# - 标准格式通常为：
#   dataset: [n, context_dim + num_actions]
#   其中前 context_dim 列是上下文特征，后 num_actions 列是每个动作的奖励。
# - 同时输出离线最优信息：
#   (opt_rewards, opt_actions)
#   - opt_rewards: 每个样本在最优动作下的奖励（或期望奖励）
#   - opt_actions: 每个样本的最优动作索引
#
# 二、全局输入来源
# - 本地文件输入：
#   - .npz（如 datasets/covertype.npz、mnist.npz、magic.npz、newsgroups.npz 等）
#   - .txt/.csv（如 stock、statlog 原始文件）
# - 在线数据输入：
#   - UCI 数据集（通过 ucimlrepo 拉取）
#   - OpenML 的 MNIST（作为本地缺失时的兜底来源）
# - 采样与预处理控制参数：
#   - num_contexts: 采样样本数
#   - shuffle_rows / shuffle_cols: 是否打乱行/列
#   - remove_underrepresented: 是否移除低频类别
#   - 任务特定参数（如 mushroom 奖励参数、stock 噪声 sigma 等）
#
# 三、统一输出约定
# - 大多数 sample_xxx_data(...) 函数返回：
#   (dataset, (opt_rewards, opt_actions))
# - 个别函数（如 sample_ag_news_data）在 return_texts=True 时额外返回文本：
#   (dataset, (opt_rewards, opt_actions), texts)
# - classification_to_bandit_problem(...) 返回分类转 Bandit 的标准表示：
#   dataset = [contexts | rewards(one-hot)]
#   以及 (opt_rewards=1 向量, opt_actions=真实类别映射后的索引)
#
# 四、关键函数的输入输出与职责
# 1) _load_bandit_npz(file_name, num_contexts, shuffle_rows)
#    输入：本地 bandit 格式 .npz 路径与采样参数
#    输出：截断/打乱后的 dataset 与 (opt_rewards, opt_actions)
#    职责：统一读取已加工好的 bandit 数据文件
#
# 2) one_hot(df, cols)
#    输入：DataFrame 与待 one-hot 的列名列表
#    输出：完成 one-hot 后的新 DataFrame
#    职责：把类别特征转为数值特征，便于后续模型训练
#
# 3) classification_to_bandit_problem(contexts, labels, num_actions=None)
#    输入：
#    - contexts: [n, d]
#    - labels: 任意标签集合（可不连续、可不从 0 开始）
#    - num_actions: 可选，动作数上限/指定值
#    输出：
#    - dataset: [n, d + K]，后 K 列为 one-hot 奖励
#    - (opt_rewards, opt_actions)
#    职责：将“分类问题”统一映射为“Bandit 奖励矩阵问题”
#
# 4) remove_underrepresented_classes(features, labels, thresh)
#    输入：特征、标签、类别占比阈值
#    输出：过滤后的 features 与 labels
#    职责：移除极少样本类别，缓解训练不稳定与极端不平衡
#
# 5) sample_xxx_data(...) 系列函数
#    输入：各数据集路径或拉取参数 + 采样/预处理参数
#    输出：统一的 bandit 数据格式
#    职责：
#    - 对接不同数据源
#    - 进行编码/清洗/抽样
#    - 产出算法可直接消费的离线 Bandit 数据
#
# 五、设计特点
# - 同时支持“本地优先、在线兜底”的数据加载策略（提高可复现性与可用性）。
# - 对分类标签做归一化映射（0..K-1），避免原始标签空洞导致索引越界。
# - 输出结构在全项目中保持一致，方便训练、评估、可视化模块复用。
# ============================================================================

import os
import io
import subprocess
import glob
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler

# 本文件位于 bandits/data/，向上三级才是项目根目录
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATASETS_DIR = os.path.join(_REPO_ROOT, "datasets")


def _load_bandit_npz(file_name: str, num_contexts: int, shuffle_rows: bool):
    """从 bandit 格式的 .npz 文件加载数据并返回标准格式。

    返回: (dataset, (opt_rewards, opt_actions))
    """
    raw = np.load(file_name)
    dataset = raw["dataset"].astype(np.float32)  # (n, context_dim + num_actions)
    opt_rewards = raw["opt_rewards"].astype(np.float32)
    opt_actions = raw["opt_actions"].astype(int)

    if shuffle_rows:
        idx = np.random.permutation(len(dataset))
        dataset = dataset[idx]
        opt_rewards = opt_rewards[idx]
        opt_actions = opt_actions[idx]

    n = min(num_contexts, len(dataset))
    return dataset[:n], (opt_rewards[:n], opt_actions[:n])


def _load_gzip_csv(file_name: str, delimiter: str = ",") -> np.ndarray:
    """Load a gzip-compressed delimited numeric file into a float32 array."""
    with subprocess.Popen(["gzip", "-dc", file_name], stdout=subprocess.PIPE, text=True) as proc:
        if proc.stdout is None:
            raise ValueError(f"Unable to read compressed file: {file_name}")
        data = np.loadtxt(proc.stdout, delimiter=delimiter, dtype=np.float32)
        return data


def _load_mnist_parquet_as_arrays(parquet_files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load HuggingFace-style MNIST parquet files into (X, y)."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to read local MNIST parquet image bytes.") from exc

    frames = [pd.read_parquet(file_name) for file_name in parquet_files]
    df = pd.concat(frames, ignore_index=True)

    contexts = []
    for image_obj in df["image"]:
        if not isinstance(image_obj, dict) or "bytes" not in image_obj:
            raise ValueError("Unexpected MNIST parquet image format; expected dict with 'bytes'.")
        image = Image.open(io.BytesIO(image_obj["bytes"]))
        contexts.append(np.asarray(image, dtype=np.float32).reshape(-1))

    labels = df["label"].to_numpy(dtype=np.int64)
    return np.vstack(contexts), labels


def one_hot(df, cols):
    """Returns one-hot encoding of DataFrame df including columns in cols."""
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
    return df


def sample_mushroom_data(num_contexts,
                         r_noeat=0,
                         r_eat_safe=5,
                         r_eat_poison_bad=-35,
                         r_eat_poison_good=5,
                         prob_poison_bad=0.5):
    """Samples bandit game from Mushroom UCI Dataset using ucimlrepo.
    Args:
        num_contexts: Number of points to sample.
        r_noeat: Reward for not eating a mushroom.
        r_eat_safe: Reward for eating a non-poisonous mushroom.
        r_eat_poison_bad: Reward for eating a poisonous mushroom if harmed.
        r_eat_poison_good: Reward for eating a poisonous mushroom if not harmed.
        prob_poison_bad: Probability of being harmed by eating a poisonous mushroom.
    Returns:
        dataset: Sampled matrix with n rows: (context, eat_reward, no_eat_reward).
        opt_vals: Vector of expected optimal (reward, action) for each context.
    """
    # Fetch mushroom dataset from UCI
    mushroom = fetch_ucirepo(id=73)
    df = mushroom.data.features
    targets = mushroom.data.targets
    
    # Combine features and targets
    df = pd.concat([df, targets], axis=1)
    
    # One-hot encode categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'string', 'category']).columns
    df = one_hot(df, categorical_columns)
    
    # Convert to numpy array and ensure float32 type
    df = df.astype(np.float32)
    
    # Sample contexts
    if num_contexts > len(df):
        num_contexts = len(df)
    ind = np.random.choice(range(len(df)), num_contexts, replace=False)

    # Get the target column (last column) for edible/poisonous indicators
    target_col = df.iloc[:, -1].values[ind]
    
    # Get contexts (all columns except the last one)
    contexts = df.iloc[ind, :-1].values
    
    no_eat_reward = r_noeat * np.ones((num_contexts, 1))
    
    # Create random poison effects
    random_poison = np.random.choice(
        [r_eat_poison_bad, r_eat_poison_good],
        p=[prob_poison_bad, 1 - prob_poison_bad],
        size=num_contexts)
    
    # Calculate eat rewards based on edible/poisonous indicators
    # Assuming 1 = edible, 0 = poisonous (adjust based on actual target encoding)
    eat_reward = r_eat_safe * target_col + np.multiply(random_poison, 1 - target_col)
    eat_reward = eat_reward.reshape((num_contexts, 1))

    # Calculate optimal expected rewards
    exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad + r_eat_poison_good * (1 - prob_poison_bad)
    opt_exp_reward = r_eat_safe * target_col + max(r_noeat, exp_eat_poison_reward) * (1 - target_col)

    if r_noeat > exp_eat_poison_reward:
        opt_actions = target_col  # indicator of edible
    else:
        opt_actions = np.ones((num_contexts, 1))

    opt_vals = (opt_exp_reward, opt_actions)

    return np.hstack((contexts, no_eat_reward, eat_reward)), opt_vals


def sample_stock_data(file_name, context_dim, num_actions, num_contexts,
                      sigma, shuffle_rows=True):
    """Samples linear bandit game from stock prices dataset."""
    contexts = np.loadtxt(file_name, skiprows=1)
    if shuffle_rows:
        np.random.shuffle(contexts)
    contexts = contexts[:num_contexts, :]

    betas = np.random.uniform(-1, 1, (context_dim, num_actions))
    betas /= np.linalg.norm(betas, axis=0)

    mean_rewards = np.dot(contexts, betas)
    noise = np.random.normal(scale=sigma, size=mean_rewards.shape)
    rewards = mean_rewards + noise

    opt_actions = np.argmax(mean_rewards, axis=1)
    opt_rewards = [mean_rewards[i, a] for i, a in enumerate(opt_actions)]
    return np.hstack((contexts, rewards)), (np.array(opt_rewards), opt_actions)


def sample_jester_data(file_name, context_dim, num_actions, num_contexts,
                       shuffle_rows=True, shuffle_cols=False):
    """Samples bandit game from (user, joke) dense subset of Jester dataset."""
    dataset = np.load(file_name)
    if shuffle_cols:
        dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
        np.random.shuffle(dataset)
    dataset = dataset[:num_contexts, :]

    assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'

    opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
    opt_rewards = np.array([dataset[i, context_dim + a] for i, a in enumerate(opt_actions)])

    return dataset, (opt_rewards, opt_actions)


# def sample_statlog_data(file_name, num_contexts, shuffle_rows=True,
#                         remove_underrepresented=False):
#     """Returns bandit problem dataset based on the UCI statlog data."""
#     data = np.loadtxt(file_name)
#     # 假设 file_name = "statlog.txt"
#     # 数据格式: 每行 = [特征1, 特征2, ..., 特征9, 标签]
#     # 前面的列是输入特征（上下文），最后一列是分类标签（1-7，代表7种不同的类别）
#     num_actions = 7 # Statlog 数据集有 7 个类别
#     if shuffle_rows:
#         np.random.shuffle(data) #打破数据的原有顺序
#     data = data[:num_contexts, :]  # 只取前num_contexts行数据
#     # 分离特征和标签
#     contexts = data[:, :-1] #提取所有行，除了最后一列的所有列作为上下文特征
#     labels = data[:, -1].astype(int) - 1 #提取所有行，最后一列作为标签，并转换为整数类型，减1使标签从0开始
#     # 移除代表性不足的类别，移除类别样本过少的数据点
#     if remove_underrepresented:
#         contexts, labels = remove_underrepresented_classes(contexts, labels)
#     return classification_to_bandit_problem(contexts, labels, num_actions)  #转换为赌博机问题，返回数据集（上下文，奖励矩阵(样本数量, 动作数量)）和（最优奖励、最优动作）

def sample_statlog_data(file_name, num_contexts, shuffle_rows=True,
                        remove_underrepresented=False):
    """Returns bandit problem dataset based on the UCI statlog data.
    Now includes feature normalization (standardization).
    """
    data = np.loadtxt(file_name)
    num_actions = 7  # Statlog 数据集有 7 个类别
    if shuffle_rows:
        np.random.shuffle(data)
    # ⚠️ 先截断，再做 normalization（避免信息泄露）
    data = data[:num_contexts, :]
    # ===== 分离特征和标签 =====
    contexts = data[:, :-1]
    labels = data[:, -1].astype(int) - 1
    # ===== 可选：移除小类 =====
    if remove_underrepresented:
        contexts, labels = remove_underrepresented_classes(contexts, labels)
    # ===== ✅ 关键：Standardization =====
    scaler = StandardScaler()
    contexts = scaler.fit_transform(contexts).astype(np.float32)
    # ===== 可选：clip（防止极端值）=====
    contexts = np.clip(contexts, -5, 5)
    # ===== 转 bandit =====
    return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_adult_data(num_contexts, shuffle_rows=True,
                      remove_underrepresented=False):
    """Returns bandit problem dataset based on the UCI adult data using ucimlrepo."""
    # Fetch adult dataset from UCI
    adult = fetch_ucirepo(id=2)
    features = adult.data.features
    targets = adult.data.targets
    
    # Combine features and targets
    data = pd.concat([features, targets], axis=1)
    
    # One-hot encode categorical columns
    categorical_columns = data.select_dtypes(include=['object', 'string', 'category']).columns
    data = one_hot(data, categorical_columns)
    
    # Convert to numpy array and ensure float32 type
    data = data.astype(np.float32).values
    
    if shuffle_rows:
        np.random.shuffle(data)
    
    if num_contexts > len(data):
        num_contexts = len(data)
    data = data[:num_contexts, :]
    
    contexts = data[:, :-1]
    labels = data[:, -1].astype(int)
    
    if remove_underrepresented:
        contexts, labels = remove_underrepresented_classes(contexts, labels)
    
    num_actions = 7
    return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_census_data(num_contexts, shuffle_rows=True,
                       remove_underrepresented=False):
    """Returns bandit problem dataset based on the UCI census data using ucimlrepo."""
    # Fetch US Census 1990 dataset from UCI
    census = fetch_ucirepo(id=116)
    features = census.data.features
    targets = census.data.targets
    
    # Combine features and targets
    data = pd.concat([features, targets], axis=1)
    
    # One-hot encode categorical columns
    categorical_columns = data.select_dtypes(include=['object', 'string', 'category']).columns
    data = one_hot(data, categorical_columns)
    
    # Convert to numpy array and ensure float32 type
    data = data.astype(np.float32).values
    
    if shuffle_rows:
        np.random.shuffle(data)
    
    if num_contexts > len(data):
        num_contexts = len(data)
    data = data[:num_contexts, :]
    
    contexts = data[:, :-1]
    labels = data[:, -1].astype(int)
    
    if remove_underrepresented:
        contexts, labels = remove_underrepresented_classes(contexts, labels)
    
    num_actions = len(np.unique(labels))
    return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_covertype_data(num_contexts, shuffle_rows=True,
                          remove_underrepresented=False):
    """Returns bandit problem dataset based on the UCI covertype data.
    优先从本地 raw 加载，其次本地 datasets/covertype.npz，最后才从 UCI 在线拉取。
    """
    raw_gz = os.path.join(_DATASETS_DIR, "raw", "covertype", "covtype.data.gz")
    local_npz = os.path.join(_DATASETS_DIR, "covertype.npz")

    if os.path.exists(raw_gz):
        data = _load_gzip_csv(raw_gz, delimiter=",")
        if shuffle_rows:
            np.random.shuffle(data)
        if num_contexts > len(data):
            num_contexts = len(data)
        data = data[:num_contexts, :]

        contexts = data[:, :-1]
        scaler = StandardScaler()
        contexts = scaler.fit_transform(contexts).astype(np.float32)
        contexts = np.clip(contexts, -5, 5)
        labels = data[:, -1].astype(int)
        if remove_underrepresented:
            contexts, labels = remove_underrepresented_classes(contexts, labels)
        num_actions = 7
        return classification_to_bandit_problem(contexts, labels, num_actions)

    if os.path.exists(local_npz):
        return _load_bandit_npz(local_npz, num_contexts, shuffle_rows)

    # 备用：在线拉取
    covertype = fetch_ucirepo(id=31)
    features = covertype.data.features
    targets = covertype.data.targets
    data = pd.concat([features, targets], axis=1)
    data = data.values

    if shuffle_rows:
        np.random.shuffle(data)
    if num_contexts > len(data):
        num_contexts = len(data)
    data = data[:num_contexts, :]

    contexts = data[:, :-1]
    scaler = StandardScaler()
    contexts = scaler.fit_transform(contexts).astype(np.float32)
    contexts = np.clip(contexts, -5, 5)
    labels = data[:, -1].astype(int)
    if remove_underrepresented:
        contexts, labels = remove_underrepresented_classes(contexts, labels)
    num_actions = 7
    return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_magic_data(num_contexts, shuffle_rows=True,
                      remove_underrepresented=False):
    """Returns bandit problem dataset based on the UCI MAGIC Gamma Telescope data.
    优先从本地 raw 加载，其次本地 datasets/magic.npz，最后才从 UCI 在线拉取。
    """
    raw_file = os.path.join(_DATASETS_DIR, "raw", "magic+gamma+telescope", "magic04.data")
    local_npz = os.path.join(_DATASETS_DIR, "magic.npz")

    if os.path.exists(raw_file):
        data = pd.read_csv(raw_file, header=None)
        contexts = data.iloc[:, :-1].to_numpy(dtype=np.float32)
        labels = pd.factorize(data.iloc[:, -1])[0].astype(int)

        if shuffle_rows:
            idx = np.random.permutation(len(contexts))
            contexts = contexts[idx]
            labels = labels[idx]

        if num_contexts > len(contexts):
            num_contexts = len(contexts)
        contexts = contexts[:num_contexts]
        scaler = StandardScaler()
        contexts = scaler.fit_transform(contexts).astype(np.float32)
        contexts = np.clip(contexts, -5, 5)
        labels = labels[:num_contexts]

        if remove_underrepresented:
            contexts, labels = remove_underrepresented_classes(contexts, labels)
        num_actions = 2
        return classification_to_bandit_problem(contexts, labels, num_actions)

    if os.path.exists(local_npz):
        return _load_bandit_npz(local_npz, num_contexts, shuffle_rows)

    # 备用：在线拉取
    magic = fetch_ucirepo(id=159)
    features = magic.data.features
    targets = magic.data.targets

    data = pd.concat([features, targets], axis=1)
    for c in data.columns:
        if not is_numeric_dtype(data[c]):
            data[c] = pd.factorize(data[c])[0]
    data = data.astype(np.float32).values

    if shuffle_rows:
        np.random.shuffle(data)
    if num_contexts > len(data):
        num_contexts = len(data)
    data = data[:num_contexts, :]

    contexts = data[:, :-1]
    scaler = StandardScaler()
    contexts = scaler.fit_transform(contexts).astype(np.float32)
    contexts = np.clip(contexts, -5, 5)
    labels = data[:, -1].astype(int)
    if remove_underrepresented:
        contexts, labels = remove_underrepresented_classes(contexts, labels)
    num_actions = 2
    return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_mnist_data(num_contexts, shuffle_rows=True,
                      remove_underrepresented=False):
    """Returns bandit problem dataset based on MNIST (OpenML mnist_784).
    优先从本地 raw 加载，其次本地 datasets/mnist.npz，最后才从 OpenML 在线拉取。
    """
    raw_pattern = os.path.join(_DATASETS_DIR, "raw", "mnist", "*.parquet")
    local_npz = os.path.join(_DATASETS_DIR, "mnist.npz")

    raw_files = sorted(glob.glob(raw_pattern))
    if raw_files:
        X, labels = _load_mnist_parquet_as_arrays(raw_files)

        if shuffle_rows:
            idx = np.random.permutation(len(X))
            X = X[idx]
            labels = labels[idx]
        if num_contexts > len(X):
            num_contexts = len(X)
        X = X[:num_contexts]
        X = X.astype(np.float32) / 255.0
        labels = labels[:num_contexts]

        if remove_underrepresented:
            X, labels = remove_underrepresented_classes(X, labels)
        num_actions = 10
        return classification_to_bandit_problem(X, labels, num_actions)

    if os.path.exists(local_npz):
        return _load_bandit_npz(local_npz, num_contexts, shuffle_rows)

    # 备用：在线拉取
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32)
    labels = y.astype(int)

    if shuffle_rows:
        idx = np.random.permutation(len(X))
        X = X[idx]
        labels = labels[idx]
    if num_contexts > len(X):
        num_contexts = len(X)
    X = X[:num_contexts]
    X = X.astype(np.float32) / 255.0
    labels = labels[:num_contexts]

    if remove_underrepresented:
        X, labels = remove_underrepresented_classes(X, labels)
    num_actions = 10
    return classification_to_bandit_problem(X, labels, num_actions)

# 将分类数据转换为赌博机问题格式，奖励为0，1
def classification_to_bandit_problem(contexts, labels, num_actions=None):
    """Converts classification data to bandit problem format."""
    n = contexts.shape[0] #样本数量
    labels = np.asarray(labels)

    # 将任意标签集合映射到连续的 0..K-1，避免原始标签从 1 开始或有空洞时越界。
    unique_labels, normalized_labels = np.unique(labels, return_inverse=True)

    # 如果未指定动作数量，则根据归一化后的标签数量自动确定动作数量
    if num_actions is None:
        num_actions = len(unique_labels)
    elif num_actions < len(unique_labels):
        raise ValueError(
            f"num_actions={num_actions} is smaller than number of unique labels={len(unique_labels)}"
        )

    rewards = np.zeros((n, num_actions)) #初始化奖励矩阵，形状为 (样本数量, 动作数量)
    rewards[np.arange(n), normalized_labels] = 1.0 #对于每个样本，正确类别的动作奖励设为1，其余为0
    opt_actions = normalized_labels #最优动作即为正确类别标签
    opt_rewards = np.ones(n) #最优奖励为1（因为正确类别的奖励为1）
    return np.hstack((contexts, rewards)), (opt_rewards, opt_actions)  # 返回数据集（横向拼接上下文和奖励矩阵）和（最优奖励、最优动作）


def safe_std(values):
    """Compute the standard deviation, returns 1.0 if values are constant."""
    std = np.std(values)
    return std if std > 0 else 1.0

# 函数：移除类别样本过少的数据点
# 样本太少的类别难以学习；避免数据不平衡问题；提高模型训练效率
def remove_underrepresented_classes(features, labels, thresh=0.0005):
    """Remove classes with very few samples."""
    unique, counts = np.unique(labels, return_counts=True) #计算每个类别的样本数量
    # unique: [0, 1, 2, 3, 4, 5, 6]
    # counts: [500, 800, 50, 300, 20, 600, 400]  (示例)
    mask = np.isin(labels, unique[counts > thresh * len(labels)]) #保留样本数量大于阈值的类别
    # 计算阈值: thresh * len(labels) = 0.0005 * 2670 = 1.335
    # 只保留样本数 > 1.335 的类别
    return features[mask], labels[mask]


def sample_newsgroups_data(file_name, num_contexts, shuffle_rows=True, return_texts=False):
    """Returns bandit problem dataset based on the 20 Newsgroups semantic data.

    The .npz file is pre-built by prepare_newsgroups.py and contains:
      dataset      – (n, context_dim + num_actions)  TF-IDF/LSA features + one-hot rewards
      opt_rewards  – (n,)
      opt_actions  – (n,)
    """
    d = np.load(file_name, allow_pickle=True)
    dataset = d['dataset'].astype(np.float32)
    opt_rewards = d['opt_rewards'].astype(np.float32)
    opt_actions = d['opt_actions'].astype(int)
    num_actions = int(d['num_actions'])
    texts = d['texts'] if 'texts' in d.files else None

    if shuffle_rows:
        idx = np.random.permutation(len(dataset))
        dataset = dataset[idx]
        opt_rewards = opt_rewards[idx]
        opt_actions = opt_actions[idx]
        if texts is not None:
            texts = texts[idx]

    if num_contexts < len(dataset):
        dataset = dataset[:num_contexts]
        opt_rewards = opt_rewards[:num_contexts]
        opt_actions = opt_actions[:num_contexts]
        if texts is not None:
            texts = texts[:num_contexts]

    if return_texts:
        return dataset, (opt_rewards, opt_actions), texts
    return dataset, (opt_rewards, opt_actions)


def sample_ag_news_data(file_name, num_contexts, shuffle_rows=True, return_texts=False):
    """Returns bandit problem dataset based on the AG News data.

    The .npz file is pre-built by prepare_ag_news.py and contains:
      dataset      – (n, context_dim + num_actions)
      opt_rewards  – (n,)
      opt_actions  – (n,)
      texts        - (n,) object array of strings
    """
    d = np.load(file_name, allow_pickle=True)
    dataset = d['dataset'].astype(np.float32)
    opt_rewards = d['opt_rewards'].astype(np.float32)
    opt_actions = d['opt_actions'].astype(int)
    texts = None
    if 'texts' in d:
        texts = d['texts']
        
    if shuffle_rows:
        idx = np.random.permutation(len(dataset))
        dataset = dataset[idx]
        opt_rewards = opt_rewards[idx]
        opt_actions = opt_actions[idx]
        if texts is not None:
            texts = texts[idx]

    if num_contexts < len(dataset):
        dataset = dataset[:num_contexts]
        opt_rewards = opt_rewards[:num_contexts]
        opt_actions = opt_actions[:num_contexts]
        if texts is not None:
            texts = texts[:num_contexts]

    if return_texts:
        return dataset, (opt_rewards, opt_actions), texts
    return dataset, (opt_rewards, opt_actions)


# def sample_statlog_shuttle_data(num_contexts, shuffle_rows=True,
#                                 remove_underrepresented=False):
#     """Returns bandit problem dataset based on the Statlog Shuttle dataset.

#     Priority:
#       1) Local raw files under datasets/raw/statlog+shuttle
#       2) Online fallback via ucimlrepo
#     """
#     local_dir = os.path.join(_DATASETS_DIR, "raw", "statlog+shuttle")
#     local_parts = []

#     tst_path = os.path.join(local_dir, "shuttle.tst")
#     trn_path = os.path.join(local_dir, "shuttle.trn")
#     trn_z_path = os.path.join(local_dir, "shuttle.trn.Z")

#     if os.path.exists(tst_path):
#         local_parts.append(np.loadtxt(tst_path, dtype=np.float32))

#     if os.path.exists(trn_path):
#         local_parts.append(np.loadtxt(trn_path, dtype=np.float32))
#     elif os.path.exists(trn_z_path):
#         # Use gzip CLI for .Z data on Unix-like systems.
#         try:
#             proc = subprocess.run(
#                 ["gzip", "-dc", trn_z_path],
#                 capture_output=True,
#                 text=True,
#                 check=True,
#             )
#             local_parts.append(np.loadtxt(io.StringIO(proc.stdout), dtype=np.float32))
#         except (subprocess.SubprocessError, FileNotFoundError, ValueError):
#             pass

#     if local_parts:
#         data = np.vstack(local_parts)
#     else:
#         # Fallback: fetch from UCI
#         shuttle = fetch_ucirepo(id=148)
#         features = shuttle.data.features
#         targets = shuttle.data.targets
#         data = pd.concat([features, targets], axis=1).values.astype(np.float32)

#     if shuffle_rows:
#         np.random.shuffle(data)

#     if num_contexts > len(data):
#         num_contexts = len(data)
#     data = data[:num_contexts, :]

#     contexts = data[:, :-1]
#     labels = data[:, -1].astype(int)

#     if remove_underrepresented:
#         contexts, labels = remove_underrepresented_classes(contexts, labels)

#     num_actions = len(np.unique(labels))
#     return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_statlog_shuttle_data(num_contexts, shuffle_rows=True, drop_time=True):
    local_dir = os.path.join(_DATASETS_DIR, "raw", "statlog+shuttle")

    trn_path = os.path.join(local_dir, "shuttle.trn")
    tst_path = os.path.join(local_dir, "shuttle.tst")

    trn = np.loadtxt(trn_path, dtype=np.float32)
    tst = np.loadtxt(tst_path, dtype=np.float32)

    data = np.vstack([trn, tst])

    if shuffle_rows:
        np.random.shuffle(data)

    if num_contexts > len(data):
        num_contexts = len(data)
    data = data[:num_contexts, :]

    if drop_time:
        contexts = data[:, 1:-1]   # 去掉第一列 time
    else:
        contexts = data[:, :-1]    # 保留全部 9 维

    labels = data[:, -1].astype(int) - 1   # 1..7 -> 0..6

    scaler = StandardScaler()
    contexts = scaler.fit_transform(contexts).astype(np.float32)
    contexts = np.clip(contexts, -5, 5)

    num_actions = 7
    return classification_to_bandit_problem(contexts, labels, num_actions)
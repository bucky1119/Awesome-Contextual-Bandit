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

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from pandas.api.types import is_numeric_dtype


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


def sample_statlog_data(file_name, num_contexts, shuffle_rows=True,
                        remove_underrepresented=False):
    """Returns bandit problem dataset based on the UCI statlog data."""
    data = np.loadtxt(file_name)
    # 假设 file_name = "statlog.txt"
    # 数据格式: 每行 = [特征1, 特征2, ..., 特征9, 标签]
    # 前面的列是输入特征（上下文），最后一列是分类标签（1-7，代表7种不同的类别）
    num_actions = 7 # Statlog 数据集有 7 个类别
    if shuffle_rows:
        np.random.shuffle(data) #打破数据的原有顺序
    data = data[:num_contexts, :]  # 只取前num_contexts行数据
    # 分离特征和标签
    contexts = data[:, :-1] #提取所有行，除了最后一列的所有列作为上下文特征
    labels = data[:, -1].astype(int) - 1 #提取所有行，最后一列作为标签，并转换为整数类型，减1使标签从0开始
    # 移除代表性不足的类别，移除类别样本过少的数据点
    if remove_underrepresented:
        contexts, labels = remove_underrepresented_classes(contexts, labels)
    return classification_to_bandit_problem(contexts, labels, num_actions)  #转换为赌博机问题，返回数据集（上下文，奖励矩阵(样本数量, 动作数量)）和（最优奖励、最优动作）


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
    
    num_actions = len(np.unique(labels))
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
    """Returns bandit problem dataset based on the UCI covertype data using ucimlrepo."""
    # Fetch covertype dataset from UCI
    covertype = fetch_ucirepo(id=31)
    features = covertype.data.features
    targets = covertype.data.targets
    
    # Combine features and targets
    data = pd.concat([features, targets], axis=1)
    
    # Convert to numpy array
    data = data.values
    
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


def sample_magic_data(num_contexts, shuffle_rows=True,
                      remove_underrepresented=False):
    """Returns bandit problem dataset based on the UCI MAGIC Gamma Telescope data."""
    # MAGIC Gamma Telescope: UCI id=159
    magic = fetch_ucirepo(id=159)
    features = magic.data.features
    targets = magic.data.targets

    data = pd.concat([features, targets], axis=1)

    # Encode non-numeric columns (including pandas StringDtype / Arrow string).
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
    labels = data[:, -1].astype(int)

    if remove_underrepresented:
        contexts, labels = remove_underrepresented_classes(contexts, labels)

    num_actions = len(np.unique(labels))
    return classification_to_bandit_problem(contexts, labels, num_actions)


def sample_mnist_data(num_contexts, shuffle_rows=True,
                      remove_underrepresented=False):
    """Returns bandit problem dataset based on MNIST (OpenML mnist_784)."""
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
    labels = labels[:num_contexts]

    if remove_underrepresented:
        X, labels = remove_underrepresented_classes(X, labels)

    num_actions = len(np.unique(labels))
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


def sample_newsgroups_data(file_name, num_contexts, shuffle_rows=True):
    """Returns bandit problem dataset based on the 20 Newsgroups semantic data.

    The .npz file is pre-built by prepare_newsgroups.py and contains:
      dataset      – (n, context_dim + num_actions)  TF-IDF/LSA features + one-hot rewards
      opt_rewards  – (n,)
      opt_actions  – (n,)
    """
    d = np.load(file_name)
    dataset = d['dataset'].astype(np.float32)
    opt_rewards = d['opt_rewards'].astype(np.float32)
    opt_actions = d['opt_actions'].astype(int)
    num_actions = int(d['num_actions'])

    if shuffle_rows:
        idx = np.random.permutation(len(dataset))
        dataset = dataset[idx]
        opt_rewards = opt_rewards[idx]
        opt_actions = opt_actions[idx]

    if num_contexts < len(dataset):
        dataset = dataset[:num_contexts]
        opt_rewards = opt_rewards[:num_contexts]
        opt_actions = opt_actions[:num_contexts]

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


def sample_statlog_shuttle_data(num_contexts, shuffle_rows=True,
                                remove_underrepresented=False):
    """Returns bandit problem dataset based on the Statlog Shuttle dataset using ucimlrepo."""
    # Fetch Statlog Shuttle dataset from UCI
    shuttle = fetch_ucirepo(id=148)
    features = shuttle.data.features
    targets = shuttle.data.targets
    
    # Combine features and targets
    data = pd.concat([features, targets], axis=1)
    
    # Convert to numpy array
    data = data.values
    
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

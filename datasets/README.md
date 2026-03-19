# Datasets 说明文档

> 自动生成于 2026-03-12 | 脚本: `prepare_datasets.py`

本目录存放所有用于上下文赌博机（Contextual Bandit）实验的数据集。

`.npz` 文件统一采用 **Bandit 格式**：

```
foo.npz
├── dataset     — float32 (n_samples, context_dim + num_actions)
│                  前 context_dim 列为上下文特征，后 num_actions 列为 one-hot 奖励矩阵
├── opt_rewards — float32 (n_samples,)   最优奖励（全为 1.0）
└── opt_actions — int32   (n_samples,)   最优动作索引（即真实类别标签）
```

---

## 数据集总览

| 文件 | 全名 | 样本数 | 特征维度 | 臂数 | 文件大小 | 任务 | 格式 |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| `statlog.npz` | Statlog (Shuttle) | 43,500 | 9 | 7 | 0.5 MB | 7分类 | npz |
| `magic.npz` | MAGIC Gamma Telescope | 19,020 | 10 | 2 | 0.7 MB | 2分类 | npz |
| `covertype.npz` | Forest Covertype | 581,012 | 54 | 7 | 13.5 MB | 7分类 | npz |
| `mnist.npz` | MNIST Handwritten Digits | 70,000 | 784 | 10 | 18.6 MB | 10分类（数字 0–9） | npz |
| `newsgroups.npz` | 20 Newsgroups（6 类，LSA 50 维） | 5,851 | 50 | 6 | 1.1 MB | 6分类 | npz |
| `ag_news.npz` | AG News（TF-IDF + SVD 100 维） | 20,000 | 15 | 4 | 2.1 MB | 4分类（World / Sports / Business / Sci&Tech） | npz |
| `ag_news_tfidf_svd100.npz` | AG News TF-IDF SVD100（备用版本） | 20,000 | 100 | 4 | 9.6 MB | 4分类 | npz |
| `adult.npz` | Adult Income（Census Income） | 32,561 | 108 | 2 | 0.6 MB | 2分类（收入 >50K / ≤50K） | npz |
| `adult.data` | adult.data | — | — | — | 4.0 MB | — | raw |
| `statlog.trn` | statlog.trn | — | — | — | 1.2 MB | — | raw |

---

## 各数据集详情

### Forest Covertype

- **文件**: `covertype.npz`
- **来源**: sklearn.datasets.fetch_covtype / UCI id=31
- **URL**: https://archive.ics.uci.edu/dataset/31/covertype
- **任务**: 7分类
- **格式**: npz
- **生成脚本**: `prepare_datasets.py`
- **样本数 (n_samples)**: 581,012
- **特征维度 (context_dim)**: 54
- **臂数 (num_actions)**: 7
- **文件大小**: 13.5 MB
- **完整性**: ✅ 通过
- **描述**: 预测科罗拉多州荒野地区的森林覆盖类型，特征包括海拔、坡度、土壤类型（one-hot）等 54 个制图变量。

#### Covertype 原始数据格式（本地 raw）

- **目录**: `datasets/raw/covertype/`
- **已下载文件**: `covtype.data.gz`, `covtype.info`
- **行格式（已核验）**: 每行 55 列，以逗号分隔
	- 前 54 列：输入特征
	- 第 55 列：类别标签（原始标签范围 1–7）
- **属性口径**:
	- 官方文档写的是 12 个属性组，但原始文件展开后共 54 列特征。
	- 其中包括 10 个数值变量、4 列 wilderness one-hot、40 列 soil type one-hot。
- **代码加载优先级**:
	1. 优先读取本地 `datasets/raw/covertype/covtype.data.gz`
	2. 若 raw 不存在，则读取 `datasets/covertype.npz`
	3. 若本地均不可用，才回退到 `ucimlrepo` 在线下载

### MNIST Handwritten Digits

- **文件**: `mnist.npz`
- **来源**: OpenML mnist_784 version=1
- **URL**: https://www.openml.org/d/554
- **任务**: 10分类（数字 0–9）
- **格式**: npz
- **生成脚本**: `prepare_datasets.py`
- **样本数 (n_samples)**: 70,000
- **特征维度 (context_dim)**: 784
- **臂数 (num_actions)**: 10
- **文件大小**: 18.6 MB
- **完整性**: ✅ 通过
- **描述**: 手写数字灰度图像，28×28 像素展平为 784 维向量，像素值归一化至 [0,1]。

#### MNIST 原始数据格式（本地 raw）

- **目录**: `datasets/raw/mnist/`
- **已下载文件**: `0000.parquet`（当前已检测到）
- **文件格式（已核验）**:
	- parquet 表包含两列：`image`, `label`
	- `label` 是数字类别标签（0–9）
	- `image` 不是直接的 784 维数组，而是一个字典对象，包含：
		- `bytes`: PNG 编码后的图像字节流
		- `path`: 路径字段（当前样本中为 `None`）
- **当前采样逻辑**:
	- 代码会先把 `image.bytes` 解码成灰度图，再展平成 784 维 float32 向量。
- **代码加载优先级**:
	1. 优先读取本地 `datasets/raw/mnist/*.parquet`
	2. 若 raw 不存在，则读取 `datasets/mnist.npz`
	3. 若本地均不可用，才回退到 OpenML 在线下载

### Statlog (Shuttle) （Asuncion & Newman,2007）

- **文件**: `statlog.npz`
- **来源**: UCI ML Repository id=148
- **URL**: https://archive.ics.uci.edu/dataset/148/statlog+shuttle
- **任务**: 7分类
- **格式**: npz
- **生成脚本**: `prepare_datasets.py`
- **样本数 (n_samples)**: 43,500
- **特征维度 (context_dim)**: 9
- **臂数 (num_actions)**: 7
- **文件大小**: 0.5 MB
- **完整性**: ✅ 通过
- **描述**: NASA 航天飞机传感器数据，9 个数值特征，用于推断飞行控制系统的工作状态（7 类）。标签由 1–7 映射为 0–6。

#### Statlog+Shuttle 原始数据格式（本地 raw）

- **目录**: `datasets/raw/statlog+shuttle/`
- **已下载文件**: `shuttle.tst`, `shuttle.trn.Z`, `Index`（以及说明文档）
- **行格式（已核验）**: 每行 10 列，以空格分隔
	- 前 9 列：数值特征（context）
	- 第 10 列：类别标签（原始标签范围 1–7）
- **口径澄清**:
	- UCI/StatLog 文档写的是 “9 attributes”，这是指 **9 个输入属性**。
	- 原始文本文件还额外包含 1 列标签，所以文件层面是 **10 列**。
	- 文档中也明确提到 “The first one being time”。若某些实现把 time 列去掉，则会变成 8 个特征；你看到“8 特征”的说法通常来自这种预处理版本。
- **样例**:
	- `55 0 81 0 -6 11 25 88 64 4`
	- `56 0 96 0 52 -4 40 44 4 4`
- **压缩文件说明**: `shuttle.trn.Z` 可通过 `gzip -dc` 或 `uncompress -c` 直接解压读取。
- **标签分布（仅 shuttle.tst）**: 1 类占比最高，2/3/6/7 类样本较少（完整训练应与 `shuttle.trn.Z` 合并后使用）。

#### 代码加载优先级（与你当前实现一致）

- 在 `bandits/data/data_sampler.py` 的 `sample_statlog_shuttle_data(...)` 中：
	1. 优先读取本地 `datasets/raw/statlog+shuttle/`（先读 `shuttle.tst`，再读 `shuttle.trn` 或 `shuttle.trn.Z`）。
	2. 仅当本地文件不可用时，才回退到 `ucimlrepo` 在线下载。

### 20 Newsgroups（6 类，LSA 50 维）

- **文件**: `newsgroups.npz`
- **来源**: sklearn.datasets.fetch_20newsgroups
- **URL**: http://qwone.com/~jason/20Newsgroups/
- **任务**: 6分类
- **格式**: npz
- **生成脚本**: `prepare_newsgroups.py`
- **样本数 (n_samples)**: 5,851
- **特征维度 (context_dim)**: 50
- **臂数 (num_actions)**: 6
- **文件大小**: 1.1 MB
- **完整性**: ✅ 通过
- **描述**: 新闻文章文本，经 TF-IDF 向量化后用 TruncatedSVD 降维至 50 维语义特征；选取 6 个主题差异较大的类别。

### MAGIC Gamma Telescope

- **文件**: `magic.npz`
- **来源**: UCI ML Repository id=159
- **URL**: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
- **任务**: 2分类
- **格式**: npz
- **生成脚本**: `bandits/data/data_sampler.py (sample_magic_data)`
- **样本数 (n_samples)**: 19,020
- **特征维度 (context_dim)**: 10
- **臂数 (num_actions)**: 2
- **文件大小**: 0.7 MB
- **完整性**: ✅ 通过
- **描述**: 大气切伦科夫望远镜模拟数据，10 个数值特征，区分 gamma 射线事件与背景强子噪声。

#### MAGIC 原始数据格式（本地 raw）

- **目录**: `datasets/raw/magic+gamma+telescope/`
- **已下载文件**: `magic04.data`, `magic04.names`
- **行格式（已核验）**: 每行 11 列，以逗号分隔
	- 前 10 列：数值特征
	- 第 11 列：类别标签，取值为 `g` 或 `h`
- **标签处理口径**:
	- 当前代码使用 `pd.factorize(...)` 将字符串标签映射为连续整数类别，再转换为 2 臂 Bandit 奖励。
- **代码加载优先级**:
	1. 优先读取本地 `datasets/raw/magic+gamma+telescope/magic04.data`
	2. 若 raw 不存在，则读取 `datasets/magic.npz`
	3. 若本地均不可用，才回退到 `ucimlrepo` 在线下载

### AG News（TF-IDF + SVD 100 维）

- **文件**: `ag_news.npz`
- **来源**: HuggingFace datasets ag_news
- **URL**: https://huggingface.co/datasets/fancyzhx/ag_news
- **任务**: 4分类（World / Sports / Business / Sci&Tech）
- **格式**: npz
- **生成脚本**: `（本地脚本生成）`
- **样本数 (n_samples)**: 20,000
- **特征维度 (context_dim)**: 15
- **臂数 (num_actions)**: 4
- **文件大小**: 2.1 MB
- **完整性**: ✅ 通过
- **描述**: 新闻标题+摘要，TF-IDF 向量化后用 SVD 降维至 100 维语义特征。

### AG News TF-IDF SVD100（备用版本）

- **文件**: `ag_news_tfidf_svd100.npz`
- **来源**: HuggingFace datasets ag_news
- **URL**: https://huggingface.co/datasets/fancyzhx/ag_news
- **任务**: 4分类
- **格式**: npz
- **生成脚本**: `（本地脚本生成）`
- **样本数 (n_samples)**: 20,000
- **特征维度 (context_dim)**: 100
- **臂数 (num_actions)**: 4
- **文件大小**: 9.6 MB
- **完整性**: ✅ 通过
- **描述**: 同 ag_news.npz，另一处理流程生成的备用版本。

### Adult Income（Census Income）

- **文件**: `adult.npz`
- **来源**: UCI ML Repository id=2
- **URL**: https://archive.ics.uci.edu/dataset/2/adult
- **任务**: 2分类（收入 >50K / ≤50K）
- **格式**: npz
- **生成脚本**: `prepare_datasets.py`
- **样本数 (n_samples)**: 32,561
- **特征维度 (context_dim)**: 108
- **臂数 (num_actions)**: 2
- **文件大小**: 0.6 MB
- **完整性**: ✅ 通过
- **描述**: 1994 年美国人口普查数据，8 个分类特征经 one-hot 编码后共约 108 维，预测年收入是否超过 $50K（0=≤50K, 1=>50K）。

### adult.data

- **文件**: `adult.data`
- **文件大小**: 4.0 MB

### statlog.trn

- **文件**: `statlog.trn`
- **文件大小**: 1.2 MB

---

## 加载方式

```python
import numpy as np

data = np.load('datasets/covertype.npz')
dataset     = data['dataset']       # (n, context_dim + num_actions)
opt_rewards = data['opt_rewards']   # (n,)
opt_actions = data['opt_actions']   # (n,)

num_actions = <已知臂数>            # 例如 covertype=7, mnist=10
context_dim = dataset.shape[1] - num_actions
contexts    = dataset[:, :context_dim]
rewards     = dataset[:, context_dim:]
```

或使用 `bandits/data/data_sampler.py` 中封装的函数按需采样。

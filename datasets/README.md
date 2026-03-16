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

# Raw Datasets 说明文档

> 自动生成于 2026-03-12 | 脚本: `download_raw_datasets.py`

本目录（`datasets/raw/`）存放**未经 Bandit 化处理的原始数据集**，
作为转换为 `.npz` bandit 格式前的数据来源。

---

## 数据集总览

| 目录 | 全名 | 总样本数 | 任务 | 格式 | 目录大小 |
| :--- | :--- | ---: | :--- | :--- | ---: |
| `20newsgroups/` | 20 Newsgroups | 18,846 | 20-class text classification (newsgroup topics) | csv | 36.4 MB |
| `ag_news/` | AG News | 127,600 | 4-class text classification (World / Sports / Business / Sci/Tech) | parquet (data/ subdirectory) | 19.8 MB |

---

## 各数据集详情

### 20 Newsgroups

- **目录**: `datasets/raw/20newsgroups/`
- **来源**: sklearn.datasets.fetch_20newsgroups
- **URL**: http://qwone.com/~jason/20Newsgroups/
- **任务**: 20-class text classification (newsgroup topics)
- **总样本数**: 18,846
- **划分**: train: 11,314, test: 7,532
- **列**: `text, label, category`
- **类别数**: 20
- **文件格式**: csv
- **备注**: remove=() — headers/footers/quotes preserved
- **下载日期**: 2026-03-12
- **目录大小**: 36.4 MB
- **文件列表**: `categories.txt`, `test.csv`, `train.csv`

### AG News

- **目录**: `datasets/raw/ag_news/`
- **来源**: HuggingFace fancyzhx/ag_news
- **URL**: https://huggingface.co/datasets/fancyzhx/ag_news
- **任务**: 4-class text classification (World / Sports / Business / Sci/Tech)
- **总样本数**: 127,600
- **划分**: test: 7,600, train: 120,000
- **列**: `text, label`
- **类别** (4): World, Sports, Business, Sci/Tech
- **文件格式**: parquet (data/ subdirectory)
- **下载日期**: 2026-03-12
- **目录大小**: 19.8 MB
- **文件列表**: `data`

---

## 加载示例

```python
import pandas as pd

# AG News / Yahoo Answers（parquet 格式）
df = pd.read_parquet('datasets/raw/ag_news/train-00000-of-00001.parquet')

# 20 Newsgroups（CSV 格式）
df = pd.read_csv('datasets/raw/20newsgroups/train.csv')
```

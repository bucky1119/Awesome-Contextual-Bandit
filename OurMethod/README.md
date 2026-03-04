# OurMethod: LLM-Enhanced Contextual Bandit with Neural LinUCB

## 概述

在现有 Deep Contextual Bandits 项目基础上，增量实现 **Prompt 生成、冻结 LLM 特征提取、特征压缩、UCB1 和 Neural LinUCB** 算法。

核心创新点：
- 通过冻结 LLM 提取最后一层潜在特征 `h_t`
- 使用轻量化 MLP 将 `h_t` 压缩为低维特征 `z_t`
- 将 `z_t` 输入 Neural LinUCB 进行在线决策
- 支持离线仿真数据冷启动增强

## 目录结构

```
OurMethod/
├── run.py                    # 可执行入口（测试 / 演示 / 实验）
├── README.md                 # 本文件
├── core/
│   ├── types.py              # 数据类型：Context / Arm / DecisionRecord
│   ├── prompt_generator.py   # Prompt 生成（角色、上下文、历史奖励）
│   ├── feature_extractor.py  # 冻结 LLM 特征提取（支持 fallback）
│   ├── feature_compressor.py # MLP 特征压缩 h_t → z_t
│   ├── ucb1.py               # UCB1 算法（含冷启动 warm_start）
│   ├── neural_linucb.py      # Neural LinUCB 算法（含冷启动）
│   ├── env_simulated.py      # 仿真环境（线性 / sigmoid 奖励）
│   └── exploration_radius.py # 自适应探索半径（UCB1 / LinUCB / 仿真增强）
├── cache/
│   └── cache_lru.py          # LRU 缓存
├── tests/
│   └── self_check.py         # 单元测试（30+ 断言）
└── logs/                     # 运行日志输出目录
```

## 模块说明

| 模块 | 功能 | 关键类/方法 |
|------|------|-------------|
| `types.py` | 统一数据结构 | `Context`, `Arm`, `DecisionRecord` |
| `prompt_generator.py` | 为 LLM 生成结构化 Prompt | `PromptGenerator.generate()`, `.generate_reward_simulation_prompt()` |
| `feature_extractor.py` | 冻结 LLM 提取 `h_t` | `FrozenLLMFeatureExtractor.extract(text)` |
| `feature_compressor.py` | 压缩 `h_t` → `z_t` | `FeatureCompressor.compress(h)` |
| `ucb1.py` | UCB1 置信区间选择 | `UCB1.action()`, `.update()`, `.warm_start()` |
| `neural_linucb.py` | Neural LinUCB 决策 | `NeuralLinUCB.action()`, `.update()`, `.warm_start()` |
| `env_simulated.py` | 仿真环境 | `SimulatedEnvironment.reward()`, `.generate_dataset()` |
| `exploration_radius.py` | 自适应探索系数 | `AdaptiveExplorationRadius.ucb1_radius()`, `.linucb_radius()` |
| `cache_lru.py` | LRU 缓存 | `LRUCache.get()`, `.put()` |

## 与现有项目的关系

- **只读复用**：通过 `from bandits.core.bandit_algorithm import BanditAlgorithm` 复用基类接口
- **不修改原文件**：所有新增代码均在 `OurMethod/` 下
- **可回滚**：删除 `OurMethod/` 目录即完全恢复原状

## 运行方式

```bash
# 进入项目根目录
cd /path/to/Deep-contextual-bandits-main

# 1. 运行单元测试
python OurMethod/run.py test

# 2. 演示 Prompt 生成
python OurMethod/run.py prompt_demo

# 3. 演示特征提取 + 压缩
python OurMethod/run.py feature_demo

# 4. 运行仿真实验 (UCB1 vs Neural LinUCB，含冷启动对比)
python OurMethod/run.py experiment --n_steps 500 --n_actions 5 --context_dim 10 --seed 42

# 5. 运行全部阶段
python OurMethod/run.py all
```

## 验证方式

```bash
# 快速验证：运行全部测试
python OurMethod/run.py test
# 预期输出：30 个测试全部 OK

# 完整验证：运行全部阶段
python OurMethod/run.py all
# 预期输出：测试通过 + Prompt 演示 + 特征演示 + 实验结果表格

# 查看实验日志
ls OurMethod/logs/
# 应包含 experiment_*.csv 和 experiment_*.json
```

## 依赖

仅使用现有项目的依赖（`numpy`, `torch`, `scipy`），无需额外安装。
如需使用真实 LLM 特征提取（非 fallback），额外安装：
```bash
pip install transformers
```

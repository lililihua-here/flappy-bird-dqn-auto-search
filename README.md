# Flappy Bird DQN 自进化超参优化系统

> 基于 Double DQN + Optuna TPE 的自动化超参搜索系统，持续逼近"最少训练帧数稳定达到 1000 分"

## 项目简介

本项目构建了一个可长期自动运行的 Flappy Bird DQN 超参搜索闭环：

1. 每个 trial 从零训练一个 DQN agent
2. 记录达到稳定 1000 分所需的训练帧数
3. 失败 trial 也返回可比较的惩罚目标值
4. TPE 利用历史结果指导下一轮参数采样
5. 支持断点续跑和 Ctrl+C 安全退出
6. 支持最优模型渲染演示

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 baseline 验证
python flappy_bird_dqn_auto_search.py --baseline-only

# Debug 模式（快速验证闭环）
python flappy_bird_dqn_auto_search.py --mode debug --max-trials 10

# Normal 模式（正式搜索）
python flappy_bird_dqn_auto_search.py --mode normal --max-trials 100

# 渲染演示搜索到的最佳 agent
python flappy_bird_dqn_auto_search.py --render --render-episodes 3
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mode` | normal | debug / normal / deep |
| `--max-trials` | 100 | 最大 trial 数 |
| `--max-trial-frames` | 按 mode | 每 trial 最大训练帧数 |
| `--history` | search_history.jsonl | JSONL 历史文件路径 |
| `--study-db` | optuna_study.db | Optuna SQLite 数据库路径 |
| `--n-startup-trials` | 30 | TPE 冷启动随机 trial 数 |
| `--baseline-only` | - | 运行 baseline trial 后退出（不搜索） |
| `--render` | - | 渲染最佳 agent 的 gameplay 画面 |
| `--render-episodes` | 1 | 渲染演示局数 |
| `--render-fps` | 60 | 渲染帧率 |
| `--checkpoint-dir` | checkpoints | 模型 checkpoint 目录 |

## 运行测试

```bash
pytest test_flappy_bird_dqn.py -v
```

## 核心指标

- **主优化目标**: `train_raw_env_frames` — 训练到稳定 1000 分消耗的训练环境帧数
- **辅助成本**: `total_raw_env_frames` — 训练 + 评估总帧数
- **稳定成功**: 20 局 greedy eval 中 >=14 局 >=1000 分，且 median >= 1000

## MVP 搜索空间（8 参数）

| 参数 | 范围 / 选项 |
|---|---|
| lr | 1e-5 ~ 3e-3 (log) |
| gamma | 0.90 ~ 0.999 |
| hidden | [64,32] / [128,64] / [256,128] |
| eps_start | 0.01 ~ 0.15 |
| eps_end | 0.001 ~ 0.02 |
| eps_decay_decision_steps | 10000 ~ 200000 |
| replay_start_size | 1000 / 5000 / 10000 |
| train_freq | 1 / 4 |

## 产物文件

| 文件 | 说明 |
|---|---|
| `search_history.jsonl` | Trial 历史（每行一个 JSON） |
| `optuna_study.db` | Optuna study 持久化（支持断点续跑） |
| `checkpoints/` | 成功 trial 的模型权重存档（用于渲染） |

## 技术栈

Python 3.11 + PyTorch + NumPy + Pygame + Optuna

## 文件说明

```
flappy-bird-dqn-auto-search/
├── flappy_bird_dqn_auto_search.py  # 主程序（单文件）
├── test_flappy_bird_dqn.py         # 单元测试和集成测试（47 个）
├── requirements.txt                # Python 依赖
├── README.md                       # 本文件
├── .gitignore
├── docs/                           # 需求文档和实施计划
├── review报告/                     # Review 报告
├── 参考论文/                       # 参考文献
└── flappy_bird_q_optimized.py      # 旧版 Q-Learning（保留参考）
```

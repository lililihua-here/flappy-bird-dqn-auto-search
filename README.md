# Flappy Bird DQN 自进化超参优化系统 (V3)

> 基于 Double DQN + Optuna TPE 的自动化超参搜索系统，持续逼近"最少训练帧数稳定达到 1000 分"

## 项目简介

本项目构建了一个可长期自动运行的 Flappy Bird DQN 超参搜索闭环：

1. 每个 trial 从零（或从父 snapshot）训练一个 DQN agent
2. 记录达到稳定 1000 分所需的训练帧数
3. 失败 trial 也返回可比较的惩罚目标值
4. TPE 利用历史结果指导下一轮参数采样
5. 支持断点续跑、same-trial 精确恢复、Ctrl+C 安全退出
6. 支持 warm-start 子 trial 和 population async 自进化搜索
7. 支持 n-step、PER、奖励比例搜索、Dueling/NoisyNet 结构扩展
8. 支持状态/奖励协议版本化与标准化实验矩阵

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 baseline 验证
python main.py --baseline-only --mode debug

# Debug 模式 TPE 搜索
python main.py --mode debug --max-trials 10

# Normal 模式搜索
python main.py --mode normal --max-trials 100

# V3 warm-start 搜索
python main.py --mode normal --max-trials 50 --search-strategy warmstart_tpe

# V3 population async 搜索
python main.py --mode normal --search-strategy population_async

# 渲染最佳 agent 的 gameplay 画面
python main.py --render --history search_history.jsonl

# 从 snapshot 精确恢复
python main.py --resume checkpoints/snapshots/snapshot_0_5000.pt

# 5-seed 最终确认
python main.py --final-confirm best_config.json

# 运行实验矩阵
python main.py --matrix structure --mode debug
```

## 运行测试

```bash
pytest -q -k "not cli_debug"
```

## 核心指标

- **主优化目标**: `train_raw_env_frames` — 训练到稳定 1000 分消耗的训练环境帧数
- **辅助成本**: `total_raw_env_frames` — 训练 + 评估总帧数
- **lineage 成本**: `lineage_train_raw_env_frames` — 继承型 trial 的完整训练帧数
- **稳定成功**: 20 局 greedy eval 中 >=14 局 >=1000 分，且 median >= 1000
- **最终确认**: 5 seed × 20 局，整体 success_rate >= 0.80，整体 median >= 1000

## V3 搜索空间（16 参数）

| 类别 | 参数 | 范围 / 选项 |
|---|---|---|
| 基础 | lr | 1e-5 ~ 3e-3 (log) |
| | gamma | 0.90 ~ 0.999 |
| | hidden | [64,32] / [128,64] / [256,128] |
| 探索 | eps_start | 0.01 ~ 0.15 |
| | eps_end | 0.001 ~ 0.02 |
| | eps_decay_decision_steps | 10000 ~ 200000 |
| Replay | replay_start_size | 1000 / 5000 / 10000 |
| | train_freq | 1 / 4 |
| 回报 | n_step | 1 / 3 / 5 |
| PER | priority | True / False |
| | per_alpha | 0.3 ~ 0.8 |
| | per_beta_start | 0.3 ~ 0.7 |
| | per_beta_train_updates | 50000 ~ 500000 |
| 奖励 | death_ratio | 5 ~ 100 |
| | reward_scale | 0.01 / 0.1 / 1.0 |
| | reward_clip | None / 10 / 100 |

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mode` | normal | debug / normal / deep |
| `--max-trials` | 100 | 最大 trial 数 |
| `--max-trial-frames` | 按 mode | 每 trial 最大训练帧数 |
| `--history` | search_history.jsonl | JSONL 历史文件路径 |
| `--study-db` | optuna_study.db | Optuna SQLite 数据库路径 |
| `--checkpoint-dir` | checkpoints | checkpoint 目录 |
| `--n-startup-trials` | 30 | TPE 冷启动随机 trial 数 |
| `--baseline-only` | - | 只跑 baseline，不搜索 |
| `--render` | - | 渲染最佳 agent 的 gameplay |
| `--render-episodes` | 1 | 渲染演示局数 |
| `--render-fps` | 60 | 渲染帧率 |
| `--report` | - | 搜索结束后生成实验报告 |
| `--resume` | - | 从 snapshot 精确恢复（trial_type=resume） |
| `--search-strategy` | tpe_fresh | tpe_fresh / warmstart_tpe / population_async |
| `--n-step` | - | 覆盖 n_step |
| `--priority` | - | 强制启用 PER |
| `--per-alpha` | 0.6 | PER alpha |
| `--per-beta-start` | 0.4 | PER beta 起始值 |
| `--death-ratio` | - | 死亡惩罚比例 |
| `--reward-scale` | - | 训练 reward 缩放系数 |
| `--matrix` | - | 运行实验矩阵 (baseline/structure/protocol/searcher) |
| `--final-confirm` | - | 5-seed 最终确认 |

## 产物文件

| 文件 | 说明 |
|---|---|
| `search_history.jsonl` | Trial 历史（每行一个 JSON） |
| `optuna_study.db` | Optuna study 持久化 |
| `checkpoints/` | checkpoint 和 snapshot |
| `checkpoints/snapshots/` | full training snapshot（支持精确恢复） |
| `summary_report.md` | 实验报告（`--report`） |
| `topk_summary.json` | Top-K 摘要 |
| `recheck_summary.json` | 最新 recheck 结果 |
| `experiment_manifest.json` | 实验元信息 |

## 技术栈

Python 3.11 + PyTorch + NumPy + Pygame + Optuna + hashlib

## 文件结构

```
flappy-bird-dqn-auto-search/
├── main.py                            # CLI 入口 + render demo
├── flappy_bird_dqn_auto_search.py     # 向后兼容 re-export shim
├── flappy_bird_env.py                 # 标准环境（支持可配置奖励参数）
├── version_utils.py                   # 共享工具
├── replay_buffer.py                   # ReplayBuffer / NStepReplayBuffer / PERBuffer / NStepPERBuffer / SumTree
├── dqn_agent.py                       # DQN / DuelingMLP / NoisyLinear / DQNAgent
├── train_eval.py                      # run_trial, greedy_eval, early_stop, objective
├── search_driver.py                   # SearchDriver, define_search_space, BASELINE_CONFIG
├── history_reporting.py               # HistoryManager, generate_summary, recheck_top_k
├── snapshot.py                        # FullTrainingSnapshot 序列化/恢复/校验
├── lineage.py                         # LineageTracker 成本追踪
├── population.py                      # PopulationController 异步群体搜索
├── state_encoder_variants.py          # StateEncoder V1/V2/V3
├── reward_protocols.py                # reward_v1_sparse / v2_ratio / v3_gap_shaping
├── experiment_matrix.py               # 标准实验矩阵 + final_confirm
├── auto_workflow.py                   # 自动化工作流
├── workflow_space.py                  # 工作流搜索空间
├── workflow_state.py                  # 工作流状态管理
├── workflow_metrics.py                # 工作流指标
├── test_agent.py                      # DQN/Agent 测试
├── test_env.py                        # 环境测试
├── test_replay.py                     # Buffer 测试
├── test_train_eval.py                 # 训练/评估测试
├── test_search.py                     # 搜索/CLI 测试
├── test_history_reporting.py          # 历史/报告测试
├── test_snapshot.py                   # snapshot 测试
├── test_population.py                 # population 测试
├── test_experiment_matrix.py          # 实验矩阵测试
├── test_state_versions.py             # 状态版本测试
├── test_reward_versions.py            # 奖励版本测试
├── test_integration.py                # 集成测试
├── test_flappy_bird_dqn.py            # 旧版兼容测试
├── test_auto_workflow.py              # 工作流测试
├── requirements.txt                   # Python 依赖
├── .gitignore
├── docs/                              # 需求文档和实施计划
├── review报告/                        # Review 报告
├── 参考论文/                          # 参考文献
└── flappy_bird_q_optimized.py         # 旧版 Q-Learning（保留参考）
```

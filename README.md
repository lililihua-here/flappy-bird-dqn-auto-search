# 🐦 Flappy Bird Q-Learning（强化学习）

> 用 **Q-Learning 强化学习**训练 AI 玩 Flappy Bird，零人工先验知识，从随机乱飞到精准穿管。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Framework](https://img.shields.io/badge/Pygame-2.5+-orange.svg)

---

## 🎮 项目简介

本项目使用**表格型 Q-Learning**（一种无模型强化学习算法），让 AI 在像素级 Flappy Bird 游戏中从零开始学习飞行策略。核心思路是将连续的游戏状态（位置、速度、管道距离）离散化为有限状态，使用 Q 表存储状态-动作价值，通过与环境交互不断更新，最终学会精准穿过管道。

### ✨ 优化亮点（相比基础 Q-Learning 版本学习速度提升 5–10 倍）

| 优化项 | 基础版 | 优化版 | 收益 |
|---|---|---|---|
| 奖励函数 | 距离负奖励 | 存活 +1，死亡 -1000，得分 +50 | 信号更清晰，避免距离奖励带来的噪声 |
| 速度分箱 | 10 格（-5 ~ +5） | 20 格（-10 ~ +30） | 覆盖完整速度范围，高速度不会被"挤"到一起 |
| 状态空间 | 3 维：Δx, Δy, v | 4 维：+「是否已过管道」标记 | 解决管道通过前后的状态混淆问题 |
| 折扣因子 γ | 0.99 | 0.95 | 更关注短期决策，适合快节奏游戏 |
| ε 衰减 | 0.9995 | 0.997 | 加速探索→利用转换，约 5 倍快 |
| 初始化 | Q=0 | Q=1.0 乐观初始化 | 鼓励 AI 在早期尝试所有动作 |

---

## 🧠 算法原理

**Q-Learning 更新公式：**

```
Q(s, a) ← Q(s, a) + α [ r + γ · max Q(s', a') - Q(s, a) ]
```

- **s** — 当前状态（小鸟与管道的位置关系 + 速度 + 是否已过管道）
- **a** — 动作（0 = 不拍 / 1 = 拍翅膀）
- **r** — 奖励信号
- **α** — 学习率，控制新旧经验的权重
- **γ** — 折扣因子，平衡短期 vs 长期奖励

**状态离散化：**

| 维度 | 范围 | 分箱数 | 说明 |
|---|---|---|---|
| Δx（水平距离） | -120 ~ +320 | 20 | 小鸟到管道的水平距离 |
| Δy（垂直距离） | -440 ~ +440 | 20 | 小鸟到管道缝隙中心的垂直距离 |
| 速度 v | -10 ~ +30 | 20 | 小鸟当前垂直速度 |
| 管道通过标记 | 0 / 1 | 2 | 是否已飞过当前管道 |

> Q 表总大小：20 × 20 × 20 × 2 × 2 = **32,000** 个状态-动作对，非常轻量。

---

## 🚀 快速开始

### 1. 环境要求

- Python **3.8+**（推荐 3.10+）
- pip（Python 包管理器）

### 2. 克隆仓库

```bash
git clone https://github.com/yuanbao6/Reinforcement-learning.git
cd Reinforcement-learning
```

### 3. 安装依赖

```bash
# 方式一：直接安装
pip install -r requirements.txt

# 方式二：在虚拟环境中安装（推荐）
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

依赖项（仅两个）：
- **pygame** ≥ 2.5.0 — 游戏渲染
- **numpy** ≥ 1.24.0 — 数组运算与 Q 表存储

### 4. 运行

```bash
python flappy_bird_q_optimized.py
```

**首次运行：** 从零开始训练（10,000 局），训练结束后自动进入测试模式展示效果。

**再次运行：** 自动检测到已保存的 Q 表文件 `q_table_opt.npz`，跳过训练直接测试。

---

## 📂 文件说明

```
Reinforcement-learning/
├── flappy_bird_q_optimized.py   # 主程序（训练 + 测试）
├── requirements.txt             # Python 依赖
├── README.md                    # 本文件
└── .gitignore
```

**训练产出（不纳入 Git，运行后生成）：**

| 文件 | 说明 |
|---|---|
| `q_table_opt.npz` | 常规存档（每 500 局保存 + 训练结束保存），含 Q 表 + ε/α/episode |
| `best_q_table_opt.npz` | 最优快照（最近 100 局平均得分最高时触发） |

---

## ⚙️ 配置参数速查

以下参数在 `flappy_bird_q_optimized.py` 顶部直接修改：

```python
# ── 游戏物理 ──
GRAVITY = 0.4          # 重力加速度
FLAP_STRENGTH = -7      # 拍翅膀的初始速度（负值=向上）
PIPE_GAP = 300          # 管道缝隙高度（越大越简单）

# ── Q-Learning 超参数 ──
EPISODES = 10000        # 训练总局数
ALPHA_START = 0.25      # 初始学习率
ALPHA_DECAY = 0.9998    # 学习率衰减系数
GAMMA = 0.95            # 折扣因子
EPSILON_DECAY = 0.997   # 探索率衰减系数（越接近 1 探索越久）

# ── 可视化 ──
TRAIN_RENDER = False    # 训练时是否显示画面（True 会很慢但能看过程）
```

### 调整建议

| 目标 | 调整方式 |
|---|---|
| 降低难度 | 增大 `PIPE_GAP`（如 350）、减小 `GRAVITY`（如 0.3） |
| 加速学习 | 减小 `EPSILON_DECAY`（如 0.995）、增大 `ALPHA_START`（如 0.3） |
| 更稳定性 | 增大 `GAMMA`（如 0.98）、减小 `ALPHA_DECAY`（如 0.9999） |
| 训练时可看画面 | 设 `TRAIN_RENDER = True` |

---

## 📊 训练效果参考

在 M 系列 MacBook Pro 上训练 10,000 局的典型效果：

- **前 1,000 局**：随机探索为主，偶尔侥幸通过 1-2 根管道
- **1,000–3,000 局**：开始学会在管道前拍翅膀，平均得分 5-15
- **3,000–6,000 局**：策略逐步收敛，平均得分 20+，最高可达 50+
- **6,000+ 局**：精细调整，稳定通过大量管道

> 💡 有预训练模型时将直接跳过训练，进入测试模式。如需重新训练，删除 `q_table_opt.npz` 即可。

---

## 🔧 常见问题

**Q: pygame 安装失败？**

```bash
# macOS
pip install pygame --pre

# Ubuntu / Debian
sudo apt-get install python3-pygame

# Windows（通常直接 pip 安装即可）
pip install pygame
```

**Q: 训练太慢？**

- 确保 `TRAIN_RENDER = False`（关闭训练渲染，这是最大的加速项）
- 减小 `EPISODES`（如 5,000 通常也能看到效果）
- 使用 PyPy 替代 CPython（通常有 2-3 倍提速）

**Q: AI 总是飞太高或太低？**

- 检查 `PIPE_GAP` 设置（过小难以学习），建议至少 250
- 降低 `GAMMA`（如 0.9）让 AI 更关注短期生存

---

## 📖 参考资料

- [Q-Learning - Wikipedia](https://en.wikipedia.org/wiki/Q-learning)
- [Reinforcement Learning: An Introduction (Sutton & Barto)](http://incompleteideas.net/book/the-book-2nd.html)
- [Pygame Documentation](https://www.pygame.org/docs/)

---

## 📄 License

MIT License — 自由使用、修改和分发。

---

🤖 *由 Q-Learning Agent 在 10,000 局训练中习得飞行技巧*

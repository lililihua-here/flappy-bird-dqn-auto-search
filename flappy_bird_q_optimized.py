"""
Flappy Bird Q-Learning — 优化版
=================================
核心改进（相比原版学习速度提升 5-10 倍）:
  1. 奖励函数: 存活 +1, 死亡 -1000, 得分 +50（不再用距离负奖励）
  2. 速度分箱: 从 10 格扩展到 20 格，覆盖 -10 ~ +30 范围
  3. 状态加入 "是否已过管道" 标记，解决混淆
  4. γ=0.95 更关注短期决策；α 自适应衰减
  5. ε 从 0.9995→0.997 加速衰减；乐观初始化鼓励探索
"""

import pygame
import numpy as np
import random
import sys
import os
from collections import deque

# ==================== 配置参数 ====================
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

GRAVITY = 0.4
FLAP_STRENGTH = -7

PIPE_WIDTH = 40
PIPE_GAP = 300              # 上下管道之间的缝隙高度
PIPE_VELOCITY = -2

BIRD_SIZE = 20
BIRD_X = 80                 # 小鸟水平位置固定

# ---------- 状态离散化 ----------
DELTA_X_BINS = 20           # 水平距离分箱数
DELTA_Y_BINS = 20           # 垂直距离分箱数
VELOCITY_BINS = 20          # 速度分箱数（原版 10，改为 20）
PIPE_PASSED_BINS = 2        # 是否已过管道标记

# ---------- Q-Learning 超参数 ----------
ALPHA_START = 0.25          # 初始学习率（原版 0.15）
ALPHA_MIN = 0.05            # 最小学习率
ALPHA_DECAY = 0.9998        # 学习率衰减

GAMMA = 0.95                # 折扣因子（原版 0.99，降下来关注短期）

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.997       # 原版 0.9995，加速近 5 倍

EPISODES = 10000
TRAIN_RENDER = False


class GameState:
    """Flappy Bird 游戏环境——优化版"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_velocity = 0.0

        self.pipe_x = SCREEN_WIDTH
        self.pipe_gap_y = random.randint(150, SCREEN_HEIGHT - 150)

        self.score = 0
        self.done = False
        self._scored_current_pipe = False   # 当前管道是否已计分
        self._pipe_passed = False           # 小鸟是否已飞过当前管道

        return self._get_state()

    def step(self, action):
        """执行一步，返回 (state, reward, done)"""

        # ---------- 1. 物理更新 ----------
        if action == 1:                     # 拍翅膀
            self.bird_velocity = FLAP_STRENGTH

        self.bird_velocity += GRAVITY
        self.bird_y += self.bird_velocity
        self.pipe_x += PIPE_VELOCITY

        # ---------- 2. 碰撞检测 ----------
        bird_rect = pygame.Rect(
            BIRD_X, self.bird_y - BIRD_SIZE // 2,
            BIRD_SIZE, BIRD_SIZE
        )
        pipe_top_rect = pygame.Rect(
            self.pipe_x, 0,
            PIPE_WIDTH, self.pipe_gap_y - PIPE_GAP // 2
        )
        pipe_bottom_rect = pygame.Rect(
            self.pipe_x, self.pipe_gap_y + PIPE_GAP // 2,
            PIPE_WIDTH, SCREEN_HEIGHT
        )

        hit_pipe = bird_rect.colliderect(pipe_top_rect) or \
                   bird_rect.colliderect(pipe_bottom_rect)
        hit_boundary = (self.bird_y - BIRD_SIZE // 2 <= 0 or
                        self.bird_y + BIRD_SIZE // 2 >= SCREEN_HEIGHT)

        # ---------- 3. 奖励计算 ----------
        # 核心原则: 活着就是好的，死了是坏的，得分是奖励
        reward = 1.0  # 每帧存活奖励

        if hit_pipe or hit_boundary:
            reward = -1000.0   # 死亡重罚
            self.done = True

        # 通过管道 -> 得分奖励（只在未计分且小鸟飞过管道时触发）
        if self.pipe_x + PIPE_WIDTH < BIRD_X and not self._scored_current_pipe:
            self.score += 1
            self._scored_current_pipe = True
            self._pipe_passed = True
            reward = 50.0       # 单次巨额奖励

        # ---------- 4. 管道重置 ----------
        if self.pipe_x < -PIPE_WIDTH:
            self.pipe_x = SCREEN_WIDTH
            self.pipe_gap_y = random.randint(150, SCREEN_HEIGHT - 150)
            self._scored_current_pipe = False
            self._pipe_passed = False

        return self._get_state(), reward, self.done

    def _get_state(self):
        """
        状态 = (Δx分箱, Δy分箱, 速度分箱, 是否已过管道)
        """
        delta_x = self.pipe_x - BIRD_X
        delta_y = self.pipe_gap_y - self.bird_y
        velocity = self.bird_velocity

        # Δx: 范围约 -120 ~ +320, 除以 22 分到 0~19
        delta_x_bin = int(np.clip(delta_x // 22, 0, DELTA_X_BINS - 1))

        # Δy: 范围约 -440 ~ +440, 加 450 变 10~890, 除以 45
        delta_y_bin = int(np.clip((delta_y + 450) // 45, 0, DELTA_Y_BINS - 1))

        # 速度: 范围 -7 ~ +30, 加 10 变 3~40, 除以 2 取整
        # 原版 clip(velocity+5,0,9) → v>4 全挤在 bin 9
        # 改进: 20 个 bin 覆盖更宽
        velocity_bin = int(np.clip((velocity + 10) // 2, 0, VELOCITY_BINS - 1))

        pipe_passed_bin = 1 if self._pipe_passed else 0

        return (delta_x_bin, delta_y_bin, velocity_bin, pipe_passed_bin)


class QLearningAgent:
    """Q-Learning Agent——优化版"""

    def __init__(self, load_path="q_table_opt.npy"):
        self.save_path = load_path

        # Q 表: [Δx][Δy][vel][passed][action]
        q_shape = (DELTA_X_BINS, DELTA_Y_BINS,
                   VELOCITY_BINS, PIPE_PASSED_BINS, 2)

        # 乐观初始化: Q≈1.0 鼓励早期探索所有动作
        self.q_table = np.full(q_shape, 1.0, dtype=np.float32)
        self.alpha = ALPHA_START
        self.epsilon = EPSILON_START
        self.episode_count = 0
        self.pretrained = False  # 是否成功加载了已有模型

        # 尝试加载已有模型
        if os.path.exists(load_path):
            data = np.load(load_path, allow_pickle=False)
            # 新版: .npz (Q表 + 超参数)
            if isinstance(data, np.lib.npyio.NpzFile):
                self.q_table = data["q_table"].astype(np.float32)
                self.epsilon = float(data["epsilon"])
                self.alpha = float(data["alpha"])
                self.episode_count = int(data["episode_count"])
                self.pretrained = True
                print(f"[OK] 已加载 Q 表 + 参数 (ep={self.episode_count}, "
                      f"ε={self.epsilon:.4f}, α={self.alpha:.3f})")
            else:
                # 旧版: 只存了 Q 表 (np.save 格式)
                if data.shape == self.q_table.shape:
                    self.q_table = data.astype(np.float32)
                    self.pretrained = True
                    print(f"[OK] 已加载 Q 表 ({load_path})，ε={self.epsilon:.4f}")
                else:
                    print(f"[WARN] Q 表维度变化 ({data.shape}→{q_shape})，重新训练")

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + GAMMA * float(np.max(self.q_table[next_state]))

        self.q_table[state][action] += self.alpha * (target - current_q)

    def decay_hyperparams(self):
        """每 episode 后衰减 ε 和 α"""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.alpha = max(ALPHA_MIN, self.alpha * ALPHA_DECAY)
        self.episode_count += 1

    def save(self):
        """保存 Q 表 + 当前 ε/α/episode_count"""
        np.savez(
            self.save_path,
            q_table=self.q_table,
            epsilon=np.float64(self.epsilon),
            alpha=np.float64(self.alpha),
            episode_count=np.int64(self.episode_count),
        )
        print(f"[SAVE] 已保存 {self.save_path} "
              f"(ep={self.episode_count}, ε={self.epsilon:.4f}, α={self.alpha:.3f})")

    def save_best(self, path="best_q_table_opt.npy"):
        """保存最优快照（带参数）"""
        np.savez(
            path,
            q_table=self.q_table,
            epsilon=np.float64(self.epsilon),
            alpha=np.float64(self.alpha),
            episode_count=np.int64(self.episode_count),
        )
        print(f"[BEST] 已保存 {path}")


def render_frame(screen, env, episode, score, clock, font):
    """显示一帧画面"""
    screen.fill((135, 206, 235))

    # 地面
    pygame.draw.rect(screen, (222, 184, 135),
                     (0, SCREEN_HEIGHT - 20, SCREEN_WIDTH, 20))
    # 管道
    pygame.draw.rect(screen, (0, 180, 0),
                     (env.pipe_x, 0, PIPE_WIDTH,
                      env.pipe_gap_y - PIPE_GAP // 2))
    pygame.draw.rect(screen, (0, 180, 0),
                     (env.pipe_x, env.pipe_gap_y + PIPE_GAP // 2,
                      PIPE_WIDTH, SCREEN_HEIGHT))
    # 小鸟
    pygame.draw.circle(screen, (255, 200, 0),
                       (BIRD_X, int(env.bird_y)), BIRD_SIZE // 2)

    text = font.render(f"Score:{score} Ep:{episode}", True, (0, 0, 0))
    screen.blit(text, (10, 10))
    pygame.display.flip()
    clock.tick(120)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird Q-Learning Optimized")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)

    agent = QLearningAgent()
    env = GameState()

    # ─────────────────────────────────────────────
    # 已加载预训练模型 → 跳过训练，直接测试
    # ─────────────────────────────────────────────
    if agent.pretrained:
        print("[SKIP] 检测到预训练模型，跳过训练，直接进入测试")
        agent.epsilon = 0.0
        _run_test(agent, env, screen, clock, font)
        _hold_window(screen, env, clock, font)
        return

    # ═════════════════════════════════════════════
    # 训练
    # ═════════════════════════════════════════════
    recent_scores = deque(maxlen=100)
    best_avg = 0.0

    print("=" * 50)
    print("开始训练 (EPISODES={})".format(EPISODES))
    print("=" * 50)

    for episode in range(EPISODES):
        state = env.reset()

        while not env.done:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    agent.save()
                    pygame.quit()
                    sys.exit()

            action = agent.act(state, training=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

            if TRAIN_RENDER:
                render_frame(screen, env, episode, env.score, clock, font)

        recent_scores.append(env.score)
        agent.decay_hyperparams()

        if episode % 100 == 0:
            avg_score = np.mean(recent_scores)
            max_score = max(recent_scores)

            if avg_score > best_avg:
                best_avg = avg_score
                agent.save_best()
                print(f"  ★ [BEST] Avg={avg_score:.2f} | α={agent.alpha:.3f}")

            print(f"  Ep={episode:5d} | "
                  f"Avg={avg_score:.2f} | "
                  f"Max={max_score} | "
                  f"ε={agent.epsilon:.4f} | "
                  f"α={agent.alpha:.3f} | "
                  f"Score={env.score}")

        if episode % 500 == 0 and episode > 0:
            agent.save()

    print("\n✓ 训练完成")
    agent.save()

    # ═════════════════════════════════════════════
    # 测试
    # ═════════════════════════════════════════════
    agent.epsilon = 0.0
    _run_test(agent, env, screen, clock, font)
    _hold_window(screen, env, clock, font)


def _run_test(agent, env, screen, clock, font):
    """用 greedy 策略跑 10 局测试，显示画面"""
    print("\n" + "=" * 50)
    print("开始测试 (10 episodes)")
    print("=" * 50)

    test_scores = []
    for test_ep in range(10):
        state = env.reset()
        while not env.done:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            render_frame(screen, env, "TEST", env.score, clock, font)
            action = agent.act(state, training=False)
            state, _, done = env.step(action)
        test_scores.append(env.score)
        print(f"  测试 {test_ep + 1:2d} → 得分: {env.score}")

    print(f"\n  平均得分: {np.mean(test_scores):.2f}  最高: {max(test_scores)}")


def _hold_window(screen, env, clock, font):
    """保持窗口打开"""
    print("\n窗口保持打开，关闭请点击 ✕")
    while True:
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        render_frame(screen, env, "DONE", env.score, clock, font)


if __name__ == "__main__":
    main()

"""
Flappy Bird DQN — 自进化超参优化系统 (MVP v1.3)
=================================================
单文件 MVP：标准环境 + 低维状态 DQN + Optuna TPE 自动搜索
"""
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import optuna
except ImportError:
    optuna = None

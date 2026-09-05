"""全局配置：随机种子、路径与超参数。

所有可调参数集中在此处，训练 / 评估 / 消融实验共用同一套配置。
"""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# ---------- 固定常量 ----------
SEED = 42                    # 随机种子，保证数据划分与初始化可复现
NUM_CLASSES = 37             # Oxford-IIIT Pet 共有 37 个宠物品种
IMAGE_SIZE = 224             # 模型输入尺寸（ResNet 标准）
RESIZE = 256                 # 先缩放到 256，再做随机/中心裁剪

BATCH_SIZE = 32
NUM_WORKERS = 0              # Kaggle 上多进程偶发 worker 错误，故设为 0
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15

# ImageNet 预训练权重的归一化统计量
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 目录
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
RUNS_DIR = Path("./runs")


def set_seed(seed: int = SEED) -> None:
    """固定 Python / NumPy / PyTorch 的随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    """一次实验所需的完整配置。

    exp_name 决定 TensorBoard 日志目录与最优权重文件名，
    方便 baseline 与消融实验互不覆盖。
    """
    exp_name: str = "pet_baseline"   # 实验名
    use_flip: bool = True            # 是否启用 RandomHorizontalFlip（消融开关）
    num_epochs: int = NUM_EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LEARNING_RATE
    seed: int = SEED
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    runs_dir: Path = RUNS_DIR

    @property
    def checkpoint_path(self) -> Path:
        """当前实验的最优权重保存路径。"""
        return self.output_dir / f"best_model_{self.exp_name}.pth"

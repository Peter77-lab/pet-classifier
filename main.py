"""实验主入口：训练 / 评估 / Grad-CAM 可视化，支持 baseline 与消融实验。

用法示例::

    # 训练 baseline（带 RandomHorizontalFlip）
    python main.py --exp-name pet_baseline

    # 消融实验（移除 RandomHorizontalFlip）
    python main.py --exp-name pet_ablation_no_flip --no-flip

    # 训练后接着在测试集评估 + 生成 Grad-CAM
    python main.py --exp-name pet_baseline --evaluate --gradcam

    # 跳过训练，仅加载已有最优权重做评估与可视化
    python main.py --exp-name pet_baseline --skip-train --evaluate --gradcam
"""

import argparse

import torch
import torch.nn as nn

from config import Config, set_seed
from data import build_dataloaders
from evaluate import evaluate
from gradcam import visualize_gradcam
from model import build_model
from train import train


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"使用设备: {device} | GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"使用设备: {device}")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet 品种分类 (ResNet-18)")
    parser.add_argument("--exp-name", type=str, default="pet_baseline",
                        help="实验名，决定权重文件名与 TensorBoard 日志目录")
    parser.add_argument("--no-flip", action="store_true",
                        help="移除 RandomHorizontalFlip（消融实验开关）")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数（默认取 config 值）")
    parser.add_argument("--batch-size", type=int, default=None, help="批大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--skip-train", action="store_true",
                        help="跳过训练，仅加载已有权重做评估/可视化")
    parser.add_argument("--evaluate", action="store_true", help="在测试集上评估")
    parser.add_argument("--gradcam", action="store_true", help="生成 Grad-CAM 可视化")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(exp_name=args.exp_name, use_flip=not args.no_flip)
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr

    set_seed(cfg.seed)
    device = get_device()

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = build_model().to(device)

    if not args.skip_train:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        train(cfg, model, train_loader, val_loader, criterion, optimizer, device)

    if args.evaluate or args.gradcam:
        model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))
        model.eval()

    if args.evaluate:
        evaluate(model, test_loader, device, cfg.exp_name, cfg.output_dir)

    if args.gradcam:
        visualize_gradcam(model, test_loader, device, cfg.output_dir)


if __name__ == "__main__":
    main()

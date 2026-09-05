"""训练与验证流程。"""

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config


def validate(model, loader: DataLoader, criterion, device) -> tuple[float, float]:
    """在给定数据集上评估，返回 (平均 loss, top-1 准确率)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            total_loss += criterion(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def train(cfg: Config, model, train_loader, val_loader, criterion, optimizer, device) -> float:
    """训练 ``num_epochs`` 轮，每轮记录 TensorBoard 指标并保存验证准确率最高的权重。

    Returns:
        最佳验证集准确率。
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.runs_dir / cfg.exp_name)

    best_val_acc = 0.0
    best_ckpt = cfg.checkpoint_path

    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss_sum = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / len(train_loader)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch[{epoch + 1:2d}/{cfg.num_epochs}] | "
              f"TrainLoss:{train_loss:.4f} | ValLoss:{val_loss:.4f} | ValAcc:{val_acc:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)

    writer.close()
    print(f"\n实验 [{cfg.exp_name}] 完成，最佳验证集准确率 = {best_val_acc:.4f}")
    return best_val_acc

"""测试集评估：Top-1 准确率 / Macro-F1 / 分类报告 / 混淆矩阵。"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model, loader: DataLoader, device) -> tuple[list, list]:
    """返回测试集全部预测标签与真实标签。"""
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        outputs = model(imgs.to(device))
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
    return all_preds, all_labels


def evaluate(model, test_loader: DataLoader, device, title: str, save_dir) -> tuple[float, float]:
    """打印分类报告、保存混淆矩阵，返回 (test_acc, macro_f1)。"""
    preds, labels = predict(model, test_loader, device)

    report = classification_report(labels, preds, output_dict=True)
    test_acc = report["accuracy"]
    macro_f1 = report["macro avg"]["f1-score"]

    print(f"========== {title} 测试集结果 ==========")
    print(f"Test Top-1 Acc: {test_acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(classification_report(labels, preds))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, cmap="Blues")
    plt.title(f"Confusion Matrix — {title}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"cm_{title}.png", dpi=150)
    plt.show()

    return test_acc, macro_f1

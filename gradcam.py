"""Grad-CAM 可视化（纯 PyTorch 手写实现）。

通过前向/反向钩子获取目标层（ResNet-18 最后一个卷积块）的激活值与梯度，
计算类别激活热力图并叠加到原图上，观察模型关注的区域。
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """对指定目标层提取 Grad-CAM 热力图。"""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """生成目标类别的 Grad-CAM 热力图（已归一化到 0-1）。"""
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, target_class].backward(retain_graph=True)

        # 对梯度做全局平均池化得到每个通道的权重，再对激活值加权求和
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        return cam[0, 0].cpu().numpy()


def visualize_gradcam(model, test_loader: DataLoader, device, save_dir) -> None:
    """各取一张预测正确/错误的样本，叠加 Grad-CAM 热力图并保存。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    cam = GradCAM(model, model.layer4[-1])

    correct, wrong = None, None
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            preds = torch.argmax(model(imgs), dim=1)
        for i in range(imgs.size(0)):
            if preds[i] == labels[i] and correct is None:
                correct = (imgs[i].cpu(), labels[i].item(), preds[i].item())
            elif preds[i] != labels[i] and wrong is None:
                wrong = (imgs[i].cpu(), labels[i].item(), preds[i].item())
            if correct is not None and wrong is not None:
                break
        if correct is not None and wrong is not None:
            break

    print("正确预测样本 Grad-CAM")
    _plot_one(model, cam, device, *correct, save_dir / "gradcam_correct.png")
    print("错误预测样本 Grad-CAM")
    _plot_one(model, cam, device, *wrong, save_dir / "gradcam_wrong.png")


def _plot_one(model, cam: GradCAM, device, img, true_label, pred_label, save_path):
    """反归一化原图并叠加热力图。target_class 使用真实标签，观察模型对真实类别的关注区域。"""
    input_tensor = img.unsqueeze(0).to(device)
    cam_map = cam.generate(input_tensor, true_label)

    # 反归一化用于显示
    img_np = img.permute(1, 2, 0).numpy()
    img_np = img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img_np = np.clip(img_np, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(img_np)
    axes[1].imshow(cam_map, cmap="jet", alpha=0.5)
    axes[1].set_title(f"Grad-CAM (True={true_label}, Pred={pred_label})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

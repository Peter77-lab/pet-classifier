# Oxford-IIIT Pet 品种分类（ResNet-18）

基于 **ResNet-18** 的宠物品种细粒度分类实验，使用 **Oxford-IIIT Pet** 数据集。
包含 baseline 训练、测试集评估、手写 Grad-CAM 可视化，以及一组「移除 RandomHorizontalFlip」的消融实验。

> 环境平台：Kaggle（Tesla T4 GPU）；深度学习框架：PyTorch。

## 任务简介

- **数据集**：[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) —— 37 个猫/狗品种，约 7,349 张图片。
- **任务**：37 类品种分类（细粒度分类）。
- **方法**：ResNet-18（ImageNet 预训练）+ 替换分类头；AdamW（lr=1e-4）训练 15 epoch。
- **数据划分**：官方仅提供 `trainval`/`test`，本实验用分层抽样（`StratifiedShuffleSplit`）把 `trainval` 划分为 **train 70% / val 15% / test 15%**，保证各品种在三个子集分布一致。

## 实验结果

| 实验 | 数据增强 | 最佳验证 Acc | 测试 Top-1 Acc | 测试 Macro-F1 |
|------|---------|:-----------:|:-------------:|:------------:|
| **Baseline (A)** | Resize + RandomCrop + **RandomHorizontalFlip** | 0.9275 | **0.9221** | **0.9214** |
| 消融 (B) | 移除 RandomHorizontalFlip | 0.9366 | 0.9149 | 0.9141 |

### 关键结论

移除 `RandomHorizontalFlip` 后，**验证准确率反而更高**（0.9366 vs 0.9275），但**测试准确率明显下降**（0.9149 vs 0.9221）。
这说明随机水平翻转作为一种数据增强，有效提升了模型的**泛化能力**、缓解了过拟合——baseline 的训练 loss 在后期已降到 0.01 左右，属于明显的过拟合，此时验证集上的差异不足以反映真实泛化性能。

## 项目结构

```
oxford-pet-classification/
├── config.py        # 全局配置：种子、超参数、路径、Config 数据类
├── data.py          # 数据加载、分层划分、DataLoader
├── model.py         # ResNet-18 模型构建
├── train.py         # 训练 / 验证流程
├── evaluate.py      # 测试集评估、混淆矩阵
├── gradcam.py       # 手写 Grad-CAM 可视化
├── main.py          # 命令行入口（训练 / 评估 / Grad-CAM / 消融）
├── requirements.txt
└── .gitignore
```

## 环境依赖

```bash
pip install -r requirements.txt
```

> 说明：Kaggle 环境已预装 PyTorch 等依赖，本地复现时按需安装 GPU 版 `torch`/`torchvision`。

## 快速开始

数据集会自动下载到 `./data/`。

```bash
# 1. 训练 baseline（带 RandomHorizontalFlip）
python main.py --exp-name pet_baseline

# 2. 训练后直接在测试集评估 + 生成 Grad-CAM
python main.py --exp-name pet_baseline --evaluate --gradcam

# 3. 消融实验（移除 RandomHorizontalFlip）
python main.py --exp-name pet_ablation_no_flip --no-flip

# 4. 跳过训练，仅加载已有最优权重做评估 / 可视化
python main.py --exp-name pet_baseline --skip-train --evaluate --gradcam
```

训练过程会：

- 使用 TensorBoard 记录 Loss / Acc，可用 `tensorboard --logdir runs/` 查看；
- 把验证准确率最高的权重保存到 `output/best_model_<exp_name>.pth`；
- 评估时输出 Top-1 Acc、Macro-F1、逐类分类报告，并保存混淆矩阵 `output/cm_<exp_name>.png`；
- Grad-CAM 分别保存一张预测正确、一张预测错误的样本热力图到 `output/`。

## 消融实验说明

| 组别 | 变体 | 说明 |
|------|------|------|
| A（baseline） | `--exp-name pet_baseline` | 完整数据增强 |
| B（消融） | `--exp-name pet_ablation_no_flip --no-flip` | 移除 `RandomHorizontalFlip` |

通过 `Config.use_flip` 开关控制，两组实验共用同一套训练流程与数据划分（`seed=42`），保证对比公平。

## 参考

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

"""数据加载与划分。

Oxford-IIIT Pet 官方只提供 ``trainval`` / ``test`` 两个 split，本实验采用
分层抽样（StratifiedShuffleSplit）把 ``trainval`` 进一步划分为
train(70%) / val(15%) / test(15%)，保证每个品种在三个子集中的比例一致。
"""

import numpy as np
import torchvision.transforms as T
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet

from config import (Config, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE,
                    NUM_WORKERS, RESIZE)

VAL_RATIO = 0.15   # 验证集占完整 trainval 的比例
TEST_RATIO = 0.15  # 测试集占完整 trainval 的比例


def build_transforms(use_flip: bool):
    """构造训练 / 验证测试两组 transform。

    训练集：Resize -> RandomCrop -> (可选 RandomHorizontalFlip) -> ToTensor -> Normalize
    验证/测试集：Resize -> CenterCrop -> ToTensor -> Normalize（不做随机增强）
    """
    train_transforms = [
        T.Resize((RESIZE, RESIZE)),
        T.RandomCrop(IMAGE_SIZE),
    ]
    if use_flip:
        train_transforms.append(T.RandomHorizontalFlip(p=0.5))
    train_transforms += [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    val_test_transforms = [
        T.Resize((RESIZE, RESIZE)),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(train_transforms), T.Compose(val_test_transforms)


def stratified_split(full_dataset, seed: int):
    """按品种对 ``trainval`` 做分层划分，返回 (train_idx, val_idx, test_idx)。

    第一次抽样：70% 训练，30% 临时集合；
    第二次抽样：临时集合再对半分，得到 15% 验证 + 15% 测试。
    """
    labels = np.array([full_dataset[i][1] for i in range(len(full_dataset))])

    temp_ratio = VAL_RATIO + TEST_RATIO
    sss_1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)
    train_idx, temp_idx = next(sss_1.split(np.arange(len(full_dataset)), labels))

    temp_labels = labels[temp_idx]
    sss_2 = StratifiedShuffleSplit(n_splits=1,
                                   test_size=TEST_RATIO / temp_ratio,
                                   random_state=seed)
    val_rel_idx, test_rel_idx = next(sss_2.split(np.arange(len(temp_idx)), temp_labels))

    return train_idx, temp_idx[val_rel_idx], temp_idx[test_rel_idx]


def build_dataloaders(cfg: Config):
    """构造 train / val / test 三个 DataLoader。

    注意：三个子集各自构造独立的 ``OxfordIIITPet`` 实例并绑定自己的 transform，
    而不是复用同一个实例后再修改 transform，避免相互影响。
    """
    train_transform, val_test_transform = build_transforms(cfg.use_flip)

    # 先下载完整数据集（不带 transform），仅用于取标签做分层抽样
    full_dataset = OxfordIIITPet(root=str(cfg.data_dir), download=True, split="trainval")
    train_idx, val_idx, test_idx = stratified_split(full_dataset, cfg.seed)

    train_set = Subset(
        OxfordIIITPet(root=str(cfg.data_dir), split="trainval", transform=train_transform),
        train_idx,
    )
    val_set = Subset(
        OxfordIIITPet(root=str(cfg.data_dir), split="trainval", transform=val_test_transform),
        val_idx,
    )
    test_set = Subset(
        OxfordIIITPet(root=str(cfg.data_dir), split="trainval", transform=val_test_transform),
        test_idx,
    )

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader

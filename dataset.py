"""
数据集工具模块
- 从单一根目录（按类别子文件夹）分层采样划分训练/验证/测试集
- 提供自定义 Dataset 类
"""
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, default_collate


def read_data_split(root: str,
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15,
                    test_ratio: float = 0.15,
                    seed: int = 42):
    """
    从根目录读取所有类别图片，按比例分层划分训练集、验证集、测试集。

    分层策略：在每个类别内部独立划分，保证每个集合中各类别比例与原数据集一致，
    避免小类别在某集合中完全缺失。

    参数:
        root: 数据根目录，每个子文件夹代表一个类别（如 anger/, happiness/ ...）
        train_ratio, val_ratio, test_ratio: 划分比例，三者之和应为 1
        seed: 随机种子，确保可复现

    返回:
        train_images, train_labels,      # 训练集路径列表和标签列表
        val_images,   val_labels,        # 验证集
        test_images,  test_labels,       # 测试集
        class_names                      # 类别名称列表（按字母序）
    """
    # 参数校验
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为 1"
    random.seed(seed)

    # 获取所有类别（按字母序固定顺序）
    class_names = sorted([d for d in os.listdir(root)
                          if os.path.isdir(os.path.join(root, d))])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # 存储各集合的路径和标签
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    for class_name in class_names:
        class_dir = os.path.join(root, class_name)
        # 收集该类下所有图片路径（支持常见格式）
        img_paths = []
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_paths.append(os.path.join(class_dir, fname))
        # 类别内随机打乱
        random.shuffle(img_paths)

        total = len(img_paths)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        class_train = img_paths[:train_end]
        class_val = img_paths[train_end:val_end]
        class_test = img_paths[val_end:]

        label = class_to_idx[class_name]
        train_paths.extend(class_train)
        train_labels.extend([label] * len(class_train))
        val_paths.extend(class_val)
        val_labels.extend([label] * len(class_val))
        test_paths.extend(class_test)
        test_labels.extend([label] * len(class_test))

    # 可选：将各集合内部整体打乱（增加随机性，避免顺序影响）
    def shuffle_pair(paths, labels):
        combined = list(zip(paths, labels))
        random.shuffle(combined)
        return zip(*combined) if combined else ([], [])

    train_paths, train_labels = shuffle_pair(train_paths, train_labels)
    val_paths, val_labels = shuffle_pair(val_paths, val_labels)
    test_paths, test_labels = shuffle_pair(test_paths, test_labels)

    return {
        'train_paths': list(train_paths),
        'train_labels': list(train_labels),
        'val_paths': list(val_paths),
        'val_labels': list(val_labels),
        'test_paths': list(test_paths),
        'test_labels': list(test_labels),
        'class_names': class_names
    }


class MyDataSet(Dataset):
    """自定义图像数据集，加载图片并应用变换。"""
    def __init__(self, images_path, images_class, transform=None):
        """
        参数:
            images_path: 图片路径列表
            images_class: 对应的标签列表
            transform: torchvision 变换（如 Resize, ToTensor, Normalize）
        """
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx]).convert('RGB')
        label = self.images_class[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        """默认的批处理拼接函数，可用于 DataLoader 的 collate_fn 参数"""
        return default_collate(batch)



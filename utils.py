import os
import sys
import random
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def read_config(config_path="config.json"):
    """读取配置文件，返回 model_config 字典。可指定 config 文件路径。"""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config['model_config']


def set_seed(seed=0):
    """固定随机种子，保证可复现（包含 CUDA 确定性设置）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 情况
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 与 deterministic 配合使用
    os.environ['PYTHONHASHSEED'] = str(seed)


class FocalLoss(nn.Module):
    """Focal Loss: 降低易分类样本的权重，聚焦难分类样本，适合类别不均衡场景。

    论文: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: 类别权重 (tensor of shape [num_classes] or None)
        gamma: 聚焦参数，越大越关注难样本 (默认 2.0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(labels, num_classes):
    """根据标签列表计算类别权重（反比于样本数），用于 CrossEntropyLoss 的 weight 参数。"""
    class_counts = torch.zeros(num_classes)
    for label in labels:
        class_counts[label] += 1
    # 避免除零
    class_counts = class_counts.clamp(min=1)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes  # 归一化使权重均值为1
    return weights


def trainer(model, optimizer, data_loader, config, epoch, criterion=None):
    """
    训练一个 epoch，返回平均损失和准确率。
    """
    device = config['device']
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        # 检查 loss 是否有限
        if not torch.isfinite(loss):
            print(f'WARNING: non-finite loss at step {step}, stopping training')
            sys.exit(1)

        optimizer.step()

        total_loss += loss.item() * images.size(0)  # 还原为总损失（未除以 batch）
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += images.size(0)

        # 更新进度条描述
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        data_loader.desc = f"[train epoch {epoch}] loss: {avg_loss:.3f}, acc: {avg_acc:.3f}"

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def evaluate(model, data_loader, config, epoch, criterion=None):
    """
    验证/测试一个 epoch，返回平均损失和准确率。
    """
    device = config['device']
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    data_loader = tqdm(data_loader, file=sys.stdout)

    with torch.no_grad():
        for step, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += images.size(0)

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            data_loader.desc = f"[eval epoch {epoch}] loss: {avg_loss:.3f}, acc: {avg_acc:.3f}"

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def compute_metrics(model, data_loader, config, class_names=None):
    """计算混淆矩阵和各类别 precision / recall / f1-score。"""
    device = config['device']
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    num_classes = config['num_class']
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(all_labels, all_preds):
        confusion[t, p] += 1

    # 各类别指标
    per_class = {}
    for c in range(num_classes):
        tp = confusion[c, c].item()
        fp = (confusion[:, c].sum() - confusion[c, c]).item()
        fn = (confusion[c, :].sum() - confusion[c, c]).item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        name = class_names[c] if class_names else str(c)
        per_class[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': confusion[c].sum().item()
        }

    # 宏平均
    macro_p = sum(v['precision'] for v in per_class.values()) / num_classes
    macro_r = sum(v['recall'] for v in per_class.values()) / num_classes
    macro_f1 = sum(v['f1'] for v in per_class.values()) / num_classes

    # 整体准确率
    correct = confusion.diag().sum().item()
    total = confusion.sum().item()
    accuracy = correct / total if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'per_class': per_class,
        'confusion': confusion
    }


def print_metrics(metrics):
    """格式化打印评估指标。"""
    print(f"\n{'='*60}")
    print(f"整体准确率: {metrics['accuracy']:.4f}")
    print(f"宏平均 Precision: {metrics['macro_precision']:.4f}  "
          f"Recall: {metrics['macro_recall']:.4f}  "
          f"F1: {metrics['macro_f1']:.4f}")
    print(f"\n{'类别':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)
    for name, m in metrics['per_class'].items():
        print(f"{name:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10}")
    print(f"\n混淆矩阵 (行=真实, 列=预测):")
    print(metrics['confusion'])
    print(f"{'='*60}\n")


# ========== 测试代码 ==========
if __name__ == "__main__":
    ...
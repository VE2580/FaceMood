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


def trainer(model, optimizer, data_loader, config, epoch):
    """
    训练一个 epoch，返回平均损失和准确率。
    """
    device = config['device']
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

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


def evaluate(model, data_loader, config, epoch):
    """
    验证/测试一个 epoch，返回平均损失和准确率。
    """
    device = config['device']
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

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


# ========== 测试代码（保持不变，仅需调整 config.json 中的字段） ==========
if __name__ == "__main__":
    # 注意：测试代码依赖的 config.json 内容可能需要根据你的新项目调整
    # 这里保持原有测试逻辑，但已确保导入完整
    ...
"""
主训练脚本
- 从单一数据根目录按比例划分训练/验证/测试集
- 训练过程中基于验证集保存最佳模型
- 训练结束后在测试集上评估最终性能
"""

import os
import sys
import torch
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation

import utils
from dataset import read_data_split, MyDataSet
from model import my_model


def main():
    # ------------------- 读取配置 -------------------
    config = utils.read_config()          # 假设 config.json 中已添加 data_root, train_ratio 等字段
    utils.set_seed(config['seed'])

    # 创建保存目录
    os.makedirs(config['save_path'], exist_ok=True)

    # ------------------- 数据准备 -------------------
    data = read_data_split(
        root=config['data_root'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio'],
        seed=config['seed']
    )
    # data 是字典，包含 keys: train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names
    train_paths, train_labels = data['train_paths'], data['train_labels']
    val_paths, val_labels = data['val_paths'], data['val_labels']
    test_paths, test_labels = data['test_paths'], data['test_labels']
    class_names = data['class_names']

    print(f"类别: {class_names}")
    print(f"训练集样本数: {len(train_paths)}")
    print(f"验证集样本数: {len(val_paths)}")
    print(f"测试集样本数: {len(test_paths)}")

    # 数据预处理
    img_size = config['img_size']
    data_transform = {
        "train": Compose([
            Resize((img_size, img_size)),
            RandomHorizontalFlip(),
            RandomRotation(15),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": Compose([
            Resize((img_size, img_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    batch_size = config['batch_size']
    num_workers = 0   # Windows 下建议 0，Linux 可调大

    # 创建 Dataset 和 DataLoader
    train_dataset = MyDataSet(train_paths, train_labels, transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=train_dataset.collate_fn
    )

    val_dataset = MyDataSet(val_paths, val_labels, transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=val_dataset.collate_fn
    )

    test_dataset = MyDataSet(test_paths, test_labels, transform=data_transform["val"])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=test_dataset.collate_fn
    )

    # ------------------- 模型、优化器、学习率调度器 -------------------
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    config['device'] = device   # 更新 config 中的 device 为实际 torch.device
    print(f"使用设备: {device}")

    model = my_model(num_classes=config['num_class']).to(device)

    # 优化器
    lr = config['learning_rate']
    weight_decay = config['weight_decay']
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=0)

    # ------------------- 训练循环 -------------------
    best_val_acc = 0.0
    best_model_path = os.path.join(config['save_path'], "best.pth")

    for epoch in range(config['epochs']):
        # 训练一个 epoch
        train_loss, train_acc = utils.trainer(model, optimizer, train_loader, config, epoch)
        # 验证集评估
        val_loss, val_acc = utils.evaluate(model, val_loader, config, epoch)

        # 打印信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{config['epochs']}]  "
              f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}  "
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}  "
              f"lr: {current_lr:.8f}")

        # 更新学习率
        scheduler.step()

        # 保存最佳验证集模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, best_model_path)
            print(f"   -> 保存最佳模型，验证集准确率: {val_acc:.4f}")

    # ------------------- 测试集最终评估 -------------------
    print("\n加载最佳模型并在测试集上评估...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = utils.evaluate(model, test_loader, config, epoch=-1)
    print(f"测试集最终结果: loss = {test_loss:.4f}, accuracy = {test_acc:.4f}")

    # 可选：保存最后一次训练的模型
    last_model_path = os.path.join(config['save_path'], f"last_epoch_{config['epochs']}.pth")
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, last_model_path)
    print(f"最后一次模型已保存至: {last_model_path}")
    print("训练完成。")


if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class my_model(nn.Module):
    """基于 ResNet-18 的表情识别模型，替换全连接层"""
    def __init__(self, num_classes=7):
        super(my_model, self).__init__()
        # 加载预训练的 ResNet-18
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # 移除原始全连接层
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        # 自定义全连接层输出为表情类别数
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)       # 提取特征
        x = self.fc(x)          # 分类
        return x








# ========== 测试代码（可粘贴到 model.py 末尾） ==========
if __name__ == "__main__":
    import torch

    print("===== 测试 model.py =====")

    # ---------- 1. 测试默认类别数 ----------
    print("\n1. 测试默认类别数 (num_classes=7)...")
    model = my_model(num_classes=7)
    # 验证模型是 nn.Module 子类
    assert isinstance(model, torch.nn.Module), "模型不是 nn.Module 的子类"
    print("   模型实例化成功 ✓")

    # ---------- 2. 测试前向传播 ----------
    print("\n2. 测试前向传播...")
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)          # 模拟 224x224 输入
    with torch.no_grad():
        y = model(x)
    # 检查输出形状
    assert y.shape == (batch_size, 7), f"期望输出形状 (2,7)，实际 {y.shape}"
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {y.shape} (期望 (2, 7)) ✓")

    # ---------- 3. 测试自定义类别数 ----------
    print("\n3. 测试自定义类别数 (num_classes=3)...")
    model3 = my_model(num_classes=3)
    with torch.no_grad():
        y3 = model3(torch.randn(1, 3, 224, 224))
    assert y3.shape == (1, 3), f"期望输出形状 (1,3)，实际 {y3.shape}"
    print(f"   输出形状: {y3.shape} (期望 (1, 3)) ✓")

    # ---------- 4. 检查全连接层输出特征数 ----------
    print("\n4. 检查全连接层输出特征数...")
    # 获取最后一个全连接层的输出特征数
    fc_out_features = model.fc.out_features
    assert fc_out_features == 7, f"fc 层输出应为 7，实际 {fc_out_features}"
    print(f"   全连接层输出特征数: {fc_out_features} ✓")

    # ---------- 5. 确认主干网络为 ResNet-18 ----------
    print("\n5. 确认骨干网络为 ResNet-18...")
    # 检查模型内部是否包含 ResNet-18 的层（例如 layer4）
    has_layer4 = False
    for name, module in model.model.named_modules():
        if name == "layer4":
            has_layer4 = True
            break
    assert has_layer4, "未找到 ResNet-18 的 layer4 模块"
    print("   骨干网络包含 layer4（ResNet-18 特有结构）✓")

    # ---------- 6. 检查预训练权重是否加载（可选） ----------
    print("\n6. 检查是否成功加载预训练权重...")
    # ResNet-18 预训练模型的 layer4.1.conv2.weight 不会完全等于随机初始化
    import torchvision.models as models
    pretrained_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # 比较某一层的权重是否一致（预训练权重应相同）
    layer_weight_pretrained = pretrained_resnet.layer4[1].conv2.weight
    layer_weight_model = model.model.layer4[1].conv2.weight
    # 如果两者差值很小，说明加载了预训练权重
    diff = torch.abs(layer_weight_pretrained - layer_weight_model).max().item()
    if diff < 1e-6:
        print(f"   预训练权重已正确加载（最大差值: {diff:.2e}）✓")
    else:
        print(f"   警告：预训练权重差异较大（最大差值: {diff:.2e}），可能权重未加载或版本不同")

    # ---------- 7. 测试模型参数总数 ----------
    print("\n7. 统计模型参数数量...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params / 1e6:.2f} M")
    print(f"   可训练参数量: {trainable_params / 1e6:.2f} M")
    # ResNet-18 通常有约 11.17 M 参数，替换全连接后变化很小
    assert 11.0e6 <= total_params <= 12.0e6, "参数数量异常，可能模型结构有误"
    print("   参数量符合预期 ✓")

    print("\n===== 所有测试通过！model.py 正常 =====")
# 第4阶段：TinyBeauty 模型训练

在此阶段，我们使用第3阶段生成的扩增数据集训练轻量级 TinyBeauty 模型。

## 目的

- 开发适合移动环境的轻量级化妆模型（目标：小于100KB）
- 通过基于 U-Net 的 CNN 模型学习化妆残差（residual）
- 通过眼线损失函数保留精细细节
- 达到高 PSNR 质量（目标：34dB 以上）

## 主要组件

### 1. TinyBeauty 模型架构

```python
class TinyBeauty(nn.Module):
    def __init__(self):
        super(TinyBeauty, self).__init__()
        
        # 编码器（3个级别的下采样）
        self.enc1 = nn.Sequential(...)  # 3 -> 16 通道
        self.pool1 = nn.MaxPool2d(...)
        
        self.enc2 = nn.Sequential(...)  # 16 -> 32 通道
        self.pool2 = nn.MaxPool2d(...)
        
        self.enc3 = nn.Sequential(...)  # 32 -> 64 通道
        self.pool3 = nn.MaxPool2d(...)
        
        # 瓶颈部分
        self.bottleneck = nn.Sequential(...)  # 64 -> 128 -> 128 通道
        
        # 解码器（3个级别的上采样 + 跳跃连接）
        self.up3 = nn.ConvTranspose2d(...)
        self.dec3 = nn.Sequential(...)
        
        self.up2 = nn.ConvTranspose2d(...)
        self.dec2 = nn.Sequential(...)
        
        self.up1 = nn.ConvTranspose2d(...)
        self.dec1 = nn.Sequential(...)
        
        # 最终输出（3通道残差）
        self.out = nn.Conv2d(16, 3, kernel_size=1)
```

### 2. 眼线损失函数

```python
class EyelinerLoss(nn.Module):
    def __init__(self):
        super(EyelinerLoss, self).__init__()
        # Sobel 滤波器（用于边缘检测）
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], ...)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], ...)
    
    def forward(self, pred, target, eye_mask=None):
        # 灰度转换
        # 通过 Sobel 滤波器计算梯度
        # 计算眼线区域的集中损失
```

## 训练方法

1. 使用残差学习方式（原始图像 + 残差 = 化妆图像）
2. L1 重构损失 + 眼线损失的组合
3. Adam 优化器 + ReduceLROnPlateau 调度器
4. 训练50个epoch，批量大小为32

## 使用方法

```bash
# TinyBeauty 模型训练
python tinybeauty_training.py
```

## 输出结果

- `tinybeauty_model/tinybeauty_best_loss.pt`: 最小损失的模型
- `tinybeauty_model/tinybeauty_best_psnr.pt`: 最大 PSNR 的模型
- `tinybeauty_model/tinybeauty_final.pt`: 最终训练的模型
- `tinybeauty_model/samples_epoch_*.png`: 每个 epoch 的结果可视化

## 质量管理

- 每个 epoch 在验证集上测量 PSNR
- 每5个 epoch 生成并可视化样本图像
- 通过 WandB 监控训练过程
- 计算并检查模型大小

## 技术细节

- **模型大小**：约 81KB（目标小于100KB）
- **图像大小**：256 x 256 像素输入/输出
- **数据集划分**：90% 训练，10% 验证
- **学习率**：2e-4（Adam）
- **损失权重**：L1 损失（1.0）+ 眼线损失（0.5）
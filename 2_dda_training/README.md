# 第2阶段：基于扩散的数据增强器 (DDA) 训练

在此阶段，我们使用第1阶段生成的5对种子数据，训练用于化妆迁移的基于扩散的数据增强器 (DDA)。

## 目的

- 基于 Stable Diffusion 模型学习化妆迁移能力
- 开发能够在有限种子数据上实现高质量化妆迁移的模型
- 包含面部细节保留和身份保持功能

## 主要组件

### 1. 残差扩散模型 (Residual Diffusion Model, RDM)

```python
class ResidualDiffusionModel(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 残差权重参数
        
    def forward(self, latents, timesteps, context, residual_input=None):
        # 基本 UNet 输出
        base_output = self.unet(latents, timesteps, context)
        
        # 应用残差（保留细节）
        if residual_input is not None:
            output = base_output.sample + self.alpha * residual_input
            return base_output._replace(sample=output)
        
        return base_output
```

### 2. 细粒度化妆模块 (Fine-Grained Makeup Module, FGMM)

```python
class FineGrainedMakeupModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 面部区域的权重参数
        self.eyes_weight = nn.Parameter(torch.tensor(1.0))
        self.lips_weight = nn.Parameter(torch.tensor(1.0))
        self.skin_weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, makeup_features, face_mask):
        # 提取面部组成部分的掩码
        eye_mask = (face_mask == 1).float()  # 眼睛区域
        lips_mask = (face_mask == 2).float() # 嘴唇区域
        skin_mask = (face_mask == 3).float() # 皮肤区域
        
        # 应用区域权重
        weighted_features = (
            self.eyes_weight * eye_mask * makeup_features +
            self.lips_weight * lips_mask * makeup_features +
            self.skin_weight * skin_mask * makeup_features
        )
        
        return weighted_features
```

## 训练方法

1. 使用 LoRA 方法对 Stable Diffusion v1.5 进行微调
2. 结合 RDM 和 FGMM 模块以保留面部细节信息
3. 实现眼线损失函数（Sobel 滤波器）以增强细节
4. 使用5对种子图像进行训练（500个 epoch）

## 使用方法

```bash
# DDA 模型训练
python dda_training.py
```

## 输出结果

- `dda_model/dda_checkpoint_epoch_{epoch}.pt`: 训练中间的检查点
- `dda_model/sample_epoch_{epoch}.png`: 训练中生成的样本图像
- `dda_model/dda_final.pt`: 最终训练的模型

## 质量管理

- 通过 WandB 跟踪训练过程中的损失
- 使用眼线损失测量细节保真度
- 每50个 epoch 生成样本图像以检查视觉质量

## 技术细节

- **学习率**: 1e-4
- **批量大小**: 1（受内存限制）
- **优化器**: Adam
- **LoRA 秩**: 16
- **训练设备**: NVIDIA V100 GPU（推荐）
# 3阶段：使用 DDA 进行数据增强

在此阶段，我们使用第2阶段训练的基于扩散的数据增强器 (DDA) 生成大量高质量的化妆图像对。

## 目的

- 使用训练好的 DDA 模型生成 4000 对化妆图像数据
- 构建具有一致风格和质量的大规模训练数据集
- 对生成数据的质量进行评估和可视化

## 主要功能

### 1. 加载 DDA 模型

```python
def load_dda_model():
    # 加载 Stable Diffusion 模型
    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    
    # 创建 RDM 和 FGMM 模型
    rdm = ResidualDiffusionModel(pipeline.unet).to(device)
    fgmm = FineGrainedMakeupModule().to(device)
    
    # 加载检查点
    checkpoint = torch.load(DDA_MODEL_PATH, map_location=device)
    rdm.load_state_dict(checkpoint['rdm_state_dict'])
    fgmm.load_state_dict(checkpoint['fgmm_state_dict'])
    pipeline.unet.load_state_dict(checkpoint['unet_state_dict'])
    
    return pipeline, rdm, fgmm
```

### 2. 数据增强流程

```python
def amplify_data(images, pipeline, rdm, fgmm, get_face_mask, batch_size=8):
    """
    使用 DDA 模型对选定图像应用化妆并生成成对数据。
    """
    rdm.eval()
    fgmm.eval()
    
    # 化妆风格 (5种 - 在第2阶段训练的风格)
    makeup_styles = ["Natural makeup", "Glamorous makeup", "Smokey eyes", 
                     "Red lip makeup", "Korean makeup"]
    
    # 对每张图像应用化妆并保存
    for batch_idx in tqdm(range(total_batches)):
        # 批处理逻辑...
        
        # 1. 加载原始图像
        # 2. 生成面部掩码
        # 3. 随机选择化妆风格
        # 4. 使用 FGMM 生成残差
        # 5. 使用 RDM 应用化妆
        # 6. 保存生成的图像
```

### 3. 质量评估

```python
def evaluate_amplified_data(num_samples=50):
    """
    评估生成数据的质量。
    主要评估指标：LPIPS, SSIM
    """
    # 随机选择样本
    sample_indices = random.sample(range(total_images), min(num_samples, total_images))
    
    # 计算 LPIPS 和 SSIM
    # 生成样本可视化
```

## 使用方法

```bash
# 执行数据增强
python data_amplification.py
```

## 输出结果

- `amplified_data/source/`: 原始人脸图像 (4000 张)
- `amplified_data/target/`: 应用化妆的图像 (4000 张)
- `amplified_data/samples_visualization.png`: 样本图像可视化

## 质量管理

- LPIPS (Learned Perceptual Image Patch Similarity): 测量原始图像与化妆图像的感知相似性
- SSIM (Structural Similarity Index Measure): 测量结构相似性
- 样本图像可视化: 对生成数据质量的视觉评估

## 技术细节

- **化妆风格**: 随机应用共 5 种风格
- **图像大小**: 512 x 512 像素
- **面部处理**: 基于 FaRL 的面部掩码生成（区分眼睛、嘴唇和皮肤区域）
- **批处理**: 为了提高内存效率，使用批处理（默认批量大小: 8）

## 注意事项

- 数据增强过程需要大量 GPU 内存，请确保有足够的 VRAM。
- 生成 4000 对图像可能需要较长时间。
- 需要 FFHQ 数据集，路径设置为 `./datasets/ffhq/`。
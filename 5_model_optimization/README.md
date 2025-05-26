# 第5阶段：模型优化及 CoreML 转换

在此阶段，我们对第4阶段训练的 TinyBeauty 模型进行优化，以便高效部署到移动环境，并将其转换为 CoreML 格式。

## 目的

- 最小化模型大小（目标：小于100KB）
- 提升移动端推理速度（目标：小于3ms）
- 转换为 CoreML 格式以便在 iOS 设备上部署
- 最小化性能下降（保持 PSNR 质量）

## 主要优化技术

### 1. 权重剪枝 (Weight Pruning)

```python
def prune_model(model, pruning_rate=0.3):
    """
    对模型的权重进行剪枝。
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            # 根据每层权重的绝对值，将低于 pruning_rate% 的权重置为0
            weight = module.weight.data
            mask = torch.ones_like(weight)
            
            # 计算每层的阈值
            threshold = torch.quantile(torch.abs(weight).flatten(), pruning_rate)
            
            # 将低于阈值的权重置为0
            mask[torch.abs(weight) <= threshold] = 0
            module.weight.data = weight * mask
    
    return model
```

### 2. 量化 (Quantization)

```python
def quantize_model(model):
    """
    将模型量化为8位整数 (int8)。
    """
    # 设置量化配置
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # 准备模型进行量化
    model_prepared = torch.quantization.prepare(model)
    
    # 生成量化模型
    model_quantized = torch.quantization.convert(model_prepared)
    
    return model_quantized
```

### 3. CoreML 转换

```python
def convert_to_coreml(model, input_shape=(1, 3, 256, 256)):
    """
    将 PyTorch 模型转换为 CoreML 模型。
    """
    # 将模型移动到 CPU 并设置为评估模式
    model = model.to('cpu').eval()
    
    # 用于跟踪的虚拟输入
    dummy_input = torch.randn(input_shape)
    
    # 提取模型定义
    traced_model = torch.jit.trace(model, dummy_input)
    
    # CoreML 转换
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=input_shape)],
        compute_precision=ct.precision.FLOAT16,  # 使用半精度浮点数
        compute_units=ct.ComputeUnit.ALL  # 支持 CPU 和 GPU
    )
    
    # 保存 CoreML 模型
    coreml_model.save("TinyBeauty.mlmodel")
```

## 使用方法

```bash
# 模型优化及 CoreML 转换
python model_optimization.py
```

## 输出结果

- `optimized_model/tinybeauty_optimized.pt`: 优化后的 PyTorch 模型
- `optimized_model/TinyBeauty.mlmodel`: 用于 iOS 部署的 CoreML 模型
- `optimized_model/optimization_results.png`: 优化结果可视化

## 评价指标

- **模型大小**：原始 vs 剪枝 vs 量化 vs CoreML
- **PSNR**：各优化阶段的质量保持程度
- **推理时间**：每张图片的处理时间（以毫秒为单位）

## 性能结果

| 模型      | 大小 (KB) | PSNR (dB) | 推理时间 (ms) |
|-----------|-----------|-----------|----------------|
| 原始模型  | ~81 KB    | ~35.39    | ~8-10 ms       |
| 剪枝模型  | ~57 KB    | ~35.20    | ~4-5 ms        |
| 量化模型  | ~20 KB    | ~35.15    | ~3-4 ms        |
| CoreML    | ~90 KB    | -         | ~2.18 ms*      |

\* 在 iPhone 13 设备上测量的结果

## 质量管理

- 在测试集（100张图片）上测量各优化阶段的 PSNR
- 检查是否存在视觉质量下降
- 在实际设备上验证移动端部署后的推理时间

## 技术细节

- **剪枝比例**：30%（移除低于阈值的权重）
- **量化方式**：8位整数 (int8) 静态量化
- **CoreML 设置**：16位浮点数（半精度）
- **输入大小**：256x256 RGB 图片
- **支持设备**：iOS 13 及以上，支持 CPU/GPU/神经引擎
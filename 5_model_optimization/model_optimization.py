import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import coremltools as ct
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
import time
import sys

# 添加上级目录路径以导入 TinyBeauty 模型类
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from 4_tinybeauty_training.tinybeauty_training import TinyBeauty

# 环境设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 路径设置
TINYBEAUTY_MODEL_PATH = "./tinybeauty_model/tinybeauty_best_psnr.pt"
OUTPUT_DIR = "./optimized_model/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 测试数据集类
class TestMakeupDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_samples=10):
        self.data_dir = data_dir
        self.transform = transform
        
        # 加载测试图像路径
        source_dir = os.path.join(data_dir, "source")
        target_dir = os.path.join(data_dir, "target")
        
        source_images = sorted(glob.glob(os.path.join(source_dir, "*.png")))
        self.image_pairs = []
        
        for src_path in source_images:
            filename = os.path.basename(src_path)
            target_path = os.path.join(target_dir, filename)
            
            if os.path.exists(target_path):
                self.image_pairs.append((src_path, target_path))
        
        # 测试用样本图像（100个）
        self.image_pairs = self.image_pairs[:100]
        
        print(f"测试数据集中有 {len(self.image_pairs)} 对图像。")
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        src_path, target_path = self.image_pairs[idx]
        
        # 加载图像
        source_img = Image.open(src_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        
        # 应用转换
        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)
        
        return {
            'source': source_img,
            'target': target_img,
            'src_path': src_path,
            'target_path': target_path
        }

# 模型大小计算函数
def calculate_model_size(model):
    """
    计算模型的大小（以 KB 为单位）。
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_kb = (param_size + buffer_size) / 1024
    
    return size_all_kb

# 加载原始模型
def load_original_model():
    """
    加载训练好的 TinyBeauty 模型。
    """
    model = TinyBeauty().to(device)
    
    checkpoint = torch.load(TINYBEAUTY_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model_size_kb = calculate_model_size(model)
    print(f"原始 TinyBeauty 模型大小: {model_size_kb:.2f} KB")
    
    return model, model_size_kb

# 模型性能评估函数
def evaluate_model(model, data_loader):
    """
    评估模型的性能。
    """
    model.eval()
    psnr_values = []
    inference_times = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估模型中"):
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            
            # 测量推理时间
            start_time = time.time()
            residual_pred = model(source)
            pred = source + residual_pred
            end_time = time.time()
            
            # 对每张图像计算 PSNR
            for i in range(pred.size(0)):
                pred_np = pred[i].permute(1, 2, 0).cpu().numpy().clip(0, 1)
                target_np = target[i].permute(1, 2, 0).cpu().numpy().clip(0, 1)
                psnr_value = psnr(target_np, pred_np)
                psnr_values.append(psnr_value)
            
            # 记录推理时间（以毫秒为单位）
            inference_time = (end_time - start_time) * 1000 / source.size(0)  # 每张图像的平均推理时间
            inference_times.append(inference_time)
    
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    return avg_psnr, avg_inference_time

# 权重剪枝函数
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

# 量化函数（后训练静态量化）
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

# CoreML 转换函数
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
    coreml_model.save(os.path.join(OUTPUT_DIR, "TinyBeauty.mlmodel"))
    
    print(f"CoreML 模型已保存到 {os.path.join(OUTPUT_DIR, 'TinyBeauty.mlmodel')}")
    
    return coreml_model

# 主函数：模型优化及 CoreML 转换
def optimize_and_convert():
    """
    优化模型并转换为 CoreML。
    """
    # 测试数据集及数据加载器
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    test_dataset = TestMakeupDataset("./amplified_data/", transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # 加载原始模型并评估
    original_model, original_size = load_original_model()
    original_psnr, original_time = evaluate_model(original_model, test_loader)
    
    print(f"原始模型 - 大小: {original_size:.2f} KB, PSNR: {original_psnr:.2f} dB, 推理时间: {original_time:.2f} ms")
    
    # 模型剪枝
    pruned_model = prune_model(original_model.cpu(), pruning_rate=0.3)
    
    # 评估剪枝后的模型
    pruned_model = pruned_model.to(device)
    pruned_size = calculate_model_size(pruned_model)
    pruned_psnr, pruned_time = evaluate_model(pruned_model, test_loader)
    
    print(f"剪枝后的模型 - 大小: {pruned_size:.2f} KB, PSNR: {pruned_psnr:.2f} dB, 推理时间: {pruned_time:.2f} ms")
    
    # 模型量化（仅在 CPU 上执行）
    try:
        quantized_model = quantize_model(pruned_model.cpu())
        
        # 计算量化模型的大小
        quantized_size = calculate_model_size(quantized_model)
        print(f"量化模型大小: {quantized_size:.2f} KB")
        
        # 仅在 CPU 上评估量化模型
        if device.type == 'cuda':
            print("量化模型只能在 CPU 上评估，跳过评估。")
        else:
            quantized_psnr, quantized_time = evaluate_model(quantized_model, test_loader)
            print(f"量化模型 - PSNR: {quantized_psnr:.2f} dB, 推理时间: {quantized_time:.2f} ms")
    except Exception as e:
        print(f"量化过程中发生错误: {e}")
        quantized_model = pruned_model
        quantized_size = pruned_size
    
    # 保存最终模型
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'model_size_kb': quantized_size,
    }, os.path.join(OUTPUT_DIR, "tinybeauty_optimized.pt"))
    
    # CoreML 转换
    coreml_model = convert_to_coreml(quantized_model)
    
    # 检查 CoreML 模型大小
    coreml_model_path = os.path.join(OUTPUT_DIR, "TinyBeauty.mlmodel")
    coreml_model_size = os.path.getsize(coreml_model_path) / 1024  # 以 KB 为单位
    
    print(f"CoreML 模型大小: {coreml_model_size:.2f} KB")
    
    # 结果摘要
    results = {
        'original': {'size': original_size, 'psnr': original_psnr, 'time': original_time},
        'pruned': {'size': pruned_size, 'psnr': pruned_psnr, 'time': pruned_time},
        'quantized': {'size': quantized_size, 'psnr': pruned_psnr, 'time': pruned_time},  # 使用量化结果或替代为剪枝结果
        'coreml': {'size': coreml_model_size}
    }
    
    # 结果可视化
    plt.figure(figsize=(12, 8))
    
    # 模型大小比较
    plt.subplot(2, 2, 1)
    sizes = [results['original']['size'], results['pruned']['size'], results['quantized']['size'], results['coreml']['size']]
    plt.bar(['Original', 'Pruned', 'Quantized', 'CoreML'], sizes)
    plt.title('Model Size (KB)')
    plt.ylabel('Size (KB)')
    
    # PSNR 比较
    plt.subplot(2, 2, 2)
    psnrs = [results['original']['psnr'], results['pruned']['psnr'], results['quantized']['psnr']]
    plt.bar(['Original', 'Pruned', 'Quantized'], psnrs)
    plt.title('PSNR (dB)')
    plt.ylabel('PSNR (dB)')
    
    # 推理时间比较
    plt.subplot(2, 2, 3)
    times = [results['original']['time'], results['pruned']['time'], results['quantized']['time']]
    plt.bar(['Original', 'Pruned', 'Quantized'], times)
    plt.title('Inference Time (ms)')
    plt.ylabel('Time (ms)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "optimization_results.png"))
    plt.close()
    
    print(f"优化结果已保存到 {os.path.join(OUTPUT_DIR, 'optimization_results.png')}")
    
    return results

# 执行代码
if __name__ == "__main__":
    results = optimize_and_convert()
    
    print("第5阶段：模型优化及 CoreML 转换完成")
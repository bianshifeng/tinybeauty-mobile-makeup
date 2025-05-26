import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import cv2
import glob
import random
import wandb
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
from datetime import datetime

# 环境设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 路径设置
AMPLIFIED_DATA_DIR = "./amplified_data/"
OUTPUT_DIR = "./tinybeauty_model/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TinyBeauty 模型定义 (基于 U-Net 的轻量级 CNN)
class TinyBeauty(nn.Module):
    def __init__(self):
        super(TinyBeauty, self).__init__()
        
        # 编码器 (下采样)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 瓶颈部分 (中间处理)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 解码器 (上采样)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 = 64 (编码器输出) + 64 (上采样输出)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 64 = 32 (编码器输出) + 32 (上采样输出)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 32 = 16 (编码器输出) + 16 (上采样输出)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 最终输出层 (预测残差)
        self.out = nn.Conv2d(16, 3, kernel_size=1)
    
    def forward(self, x):
        # 编码器路径
        enc1_out = self.enc1(x)
        pool1_out = self.pool1(enc1_out)
        
        enc2_out = self.enc2(pool1_out)
        pool2_out = self.pool2(enc2_out)
        
        enc3_out = self.enc3(pool2_out)
        pool3_out = self.pool3(enc3_out)
        
        # 瓶颈部分
        bottleneck_out = self.bottleneck(pool3_out)
        
        # 解码器路径 (包含跳跃连接)
        up3_out = self.up3(bottleneck_out)
        cat3_out = torch.cat([up3_out, enc3_out], dim=1)
        dec3_out = self.dec3(cat3_out)
        
        up2_out = self.up2(dec3_out)
        cat2_out = torch.cat([up2_out, enc2_out], dim=1)
        dec2_out = self.dec2(cat2_out)
        
        up1_out = self.up1(dec2_out)
        cat1_out = torch.cat([up1_out, enc1_out], dim=1)
        dec1_out = self.dec1(cat1_out)
        
        # 最终输出 (残差)
        residual = self.out(dec1_out)
        
        # 模型仅返回残差
        return residual

# 眼线损失函数定义
class EyelinerLoss(nn.Module):
    def __init__(self):
        super(EyelinerLoss, self).__init__()
        # Sobel 滤波器核 (水平, 垂直)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    
    def forward(self, pred, target, eye_mask=None):
        # 灰度转换 (RGB -> 灰度)
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        # 应用 Sobel 滤波器进行边缘检测
        pred_grad_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        
        # 计算梯度幅值
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
        
        # 计算眼线区域的损失
        if eye_mask is not None:
            # 如果提供了掩码，仅考虑该区域
            loss = F.mse_loss(pred_grad_mag * eye_mask, target_grad_mag * eye_mask)
        else:
            # 对整个图像计算
            loss = F.mse_loss(pred_grad_mag, target_grad_mag)
        
        return loss

# 数据集类
class MakeupPairDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        
        # 构建原始图像和化妆图像路径对
        source_dir = os.path.join(data_dir, "source")
        target_dir = os.path.join(data_dir, "target")
        
        source_images = sorted(glob.glob(os.path.join(source_dir, "*.png")))
        self.image_pairs = []
        
        for src_path in source_images:
            filename = os.path.basename(src_path)
            target_path = os.path.join(target_dir, filename)
            
            if os.path.exists(target_path):
                self.image_pairs.append((src_path, target_path))
        
        # 训练/验证集划分
        if split in ['train', 'val']:
            train_pairs, val_pairs = train_test_split(
                self.image_pairs, test_size=0.1, random_state=42
            )
            self.image_pairs = train_pairs if split == 'train' else val_pairs
        
        print(f"{split} 数据集中有 {len(self.image_pairs)} 对图像。")
    
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
        
        # 计算残差 (目标 - 原始)
        residual = target_img - source_img
        
        # 检测眼部区域（此处为简化实现）
        # 实际实现中可使用面部关键点检测
        eye_mask = torch.zeros_like(source_img[0:1])
        h, w = eye_mask.shape[1:]
        
        # 眼部区域（假设为顶部1/3区域的中央部分）
        eye_y = h // 3
        eye_h = h // 10
        eye_mask[:, eye_y:eye_y+eye_h, w//4:3*w//4] = 1.0
        
        return {
            'source': source_img,
            'target': target_img,
            'residual': residual,
            'eye_mask': eye_mask
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

# 训练循环
def train_tinybeauty(num_epochs=50, batch_size=32, learning_rate=2e-4):
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 创建数据集和数据加载器
    train_dataset = MakeupPairDataset(AMPLIFIED_DATA_DIR, transform, split='train')
    val_dataset = MakeupPairDataset(AMPLIFIED_DATA_DIR, transform, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 初始化模型
    model = TinyBeauty().to(device)
    
    # 损失函数
    l1_loss = nn.L1Loss()
    eyeliner_loss = EyelinerLoss()
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # 初始化 wandb
    wandb.init(project="tinybeauty-training", name="tinybeauty-v1")
    
    # 计算并记录模型大小
    model_size_kb = calculate_model_size(model)
    print(f"TinyBeauty 模型大小: {model_size_kb:.2f} KB")
    wandb.log({"model_size_kb": model_size_kb})
    
    # 保存最佳模型
    best_val_loss = float('inf')
    best_val_psnr = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_samples = 0
        
        # 训练阶段
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as t:
            for batch in t:
                source = batch['source'].to(device)
                target = batch['target'].to(device)
                residual_gt = batch['residual'].to(device)
                eye_mask = batch['eye_mask'].to(device)
                
                # 前向传播
                residual_pred = model(source)
                
                # 最终预测 (原始 + 残差)
                pred = source + residual_pred
                
                # 计算损失
                l1_loss_value = l1_loss(residual_pred, residual_gt)
                eyeliner_loss_value = eyeliner_loss(pred, target, eye_mask)
                
                # 加权后的最终损失
                loss = l1_loss_value + 0.5 * eyeliner_loss_value
                
                # 反向传播及优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 更新统计数据
                batch_size = source.size(0)
                train_loss += loss.item() * batch_size
                train_samples += batch_size
                
                # 更新 tqdm 进度条显示
                t.set_postfix(loss=f"{loss.item():.4f}")
        
        # 每个 epoch 的平均训练损失
        avg_train_loss = train_loss / train_samples
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_psnr_sum = 0
        val_samples = 0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") as t:
                for batch in t:
                    source = batch['source'].to(device)
                    target = batch['target'].to(device)
                    residual_gt = batch['residual'].to(device)
                    eye_mask = batch['eye_mask'].to(device)
                    
                    # 前向传播
                    residual_pred = model(source)
                    
                    # 最终预测 (原始 + 残差)
                    pred = source + residual_pred
                    
                    # 计算损失
                    l1_loss_value = l1_loss(residual_pred, residual_gt)
                    eyeliner_loss_value = eyeliner_loss(pred, target, eye_mask)
                    
                    # 加权后的最终损失
                    loss = l1_loss_value + 0.5 * eyeliner_loss_value
                    
                    # 计算 PSNR
                    for i in range(pred.size(0)):
                        pred_np = pred[i].permute(1, 2, 0).cpu().numpy().clip(0, 1)
                        target_np = target[i].permute(1, 2, 0).cpu().numpy().clip(0, 1)
                        psnr_value = psnr(target_np, pred_np)
                        val_psnr_sum += psnr_value
                    
                    # 更新统计数据
                    batch_size = source.size(0)
                    val_loss += loss.item() * batch_size
                    val_samples += batch_size
                    
                    # 更新 tqdm 进度条显示
                    t.set_postfix(loss=f"{loss.item():.4f}")
        
        # 每个 epoch 的平均验证损失和 PSNR
        avg_val_loss = val_loss / val_samples
        avg_val_psnr = val_psnr_sum / val_samples
        
        # 更新学习率调度器
        scheduler.step(avg_val_loss)
        
        # 输出结果
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val PSNR: {avg_val_psnr:.2f} dB")
        
        # 记录到 wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_psnr": avg_val_psnr,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_psnr': avg_val_psnr,
            }, os.path.join(OUTPUT_DIR, "tinybeauty_best_loss.pt"))
            print(f"Best model (loss) saved at epoch {epoch+1}")
        
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_psnr': avg_val_psnr,
            }, os.path.join(OUTPUT_DIR, "tinybeauty_best_psnr.pt"))
            print(f"Best model (PSNR) saved at epoch {epoch+1}")
        
        # 可视化样本图像（每个 epoch）
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                # 从验证集中选择一些样本
                vis_batch = next(iter(val_loader))
                source = vis_batch['source'][:5].to(device)  # 仅使用5个样本
                target = vis_batch['target'][:5].to(device)
                
                # 预测
                residual_pred = model(source)
                pred = source + residual_pred
                
                # 创建网格图像
                vis_images = []
                for i in range(source.size(0)):
                    vis_images.append(source[i])
                    vis_images.append(pred[i])
                    vis_images.append(target[i])
                
                grid = make_grid(vis_images, nrow=3, normalize=True)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                
                # 保存图像
                plt.figure(figsize=(15, 10))
                plt.imshow(grid_np)
                plt.axis('off')
                plt.title(f"Epoch {epoch+1} - Source | Prediction | Target")
                plt.savefig(os.path.join(OUTPUT_DIR, f"samples_epoch_{epoch+1}.png"))
                plt.close()
                
                # 记录到 wandb
                wandb.log({
                    "samples": wandb.Image(os.path.join(OUTPUT_DIR, f"samples_epoch_{epoch+1}.png"))
                })
    
    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'val_psnr': avg_val_psnr,
    }, os.path.join(OUTPUT_DIR, "tinybeauty_final.pt"))
    
    # 结束 wandb
    wandb.finish()
    
    print(f"最终验证 PSNR: {avg_val_psnr:.2f} dB")
    print(f"最佳验证 PSNR: {best_val_psnr:.2f} dB")
    print(f"TinyBeauty 模型大小: {model_size_kb:.2f} KB")
    
    return model, best_val_psnr, model_size_kb

# 执行代码
if __name__ == "__main__":
    # 训练 TinyBeauty 模型
    model, best_psnr, model_size = train_tinybeauty(num_epochs=50, batch_size=32, learning_rate=2e-4)
    
    print("第4阶段：TinyBeauty 模型训练完成")

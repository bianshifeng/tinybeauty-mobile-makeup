import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
from scipy import stats
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import cv2
from matplotlib.lines import Line2D
from datetime import datetime
import glob
from collections import Counter

# 环境设置
OUTPUT_DIR = "./user_study/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 为更真实的用户研究结果模拟设置的常量
TOTAL_PARTICIPANTS = 100  # 论文中提到的参与者数量
EVALUATION_METRICS = [
    "化妆质量",
    "身份保留",
    "细节保真度",
    "整体满意度"
]
MAKEUP_METHODS = [
    "TinyBeauty", 
    "BeautyGAN", 
    "EleGANt", 
    "PSGAN", 
    "BeautyDiffusion"
]

# 测试图像数据集定义
class TestImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_samples=10):
        self.transform = transform
        
        # 加载原始图像
        source_dir = os.path.join(data_dir, "source")
        source_images = sorted(glob.glob(os.path.join(source_dir, "*.png")))
        
        # 随机采样（用于用户研究的图像数量）
        if len(source_images) > num_samples:
            self.image_paths = random.sample(source_images, num_samples)
        else:
            self.image_paths = source_images
        
        print(f"用户研究中使用了 {len(self.image_paths)} 张图像。")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'path': img_path
        }

# 应用多种化妆方法生成结果（模拟）
def apply_makeup_methods(image, methods=MAKEUP_METHODS):
    """
    对输入图像应用多种化妆方法。
    在实际实现中，每种方法会使用训练好的模型，
    但在这里为了模拟，仅应用简单的变换。
    """
    results = {}
    
    # 将图像转换为 NumPy 数组
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = np.array(image)
    
    # 保存原始图像
    results['Original'] = image_np
    
    # 对每种化妆方法应用变换（模拟而非实际模型）
    for method in methods:
        # 模拟化妆效果
        if method == "TinyBeauty":
            # TinyBeauty 的效果：自然化妆，保留细节
            # 略微增加亮度和饱和度，改善肤色
            makeup = image_np.copy()
            # 面部区域（简单的中心椭圆形遮罩）
            h, w = makeup.shape[:2]
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(face_mask, (w//2, h//2), (w//3, h//2), 0, 0, 360, 255, -1)
            
            # 改善肤色（稍微提亮并增加红色）
            face_area = makeup[face_mask > 0]
            face_area = np.clip(face_area * 1.05, 0, 255).astype(np.uint8)  # 增加亮度
            face_area[:, 1] = np.clip(face_area[:, 1] * 1.02, 0, 255).astype(np.uint8)  # 略微增加绿色通道
            face_area[:, 2] = np.clip(face_area[:, 2] * 1.08, 0, 255).astype(np.uint8)  # 增加红色通道
            makeup[face_mask > 0] = face_area
            
            # 唇部区域（简单的下部中心椭圆形）
            lip_mask = np.zeros((h, w), dtype=np.uint8)
            lip_y = int(h * 0.65)
            lip_h = int(h * 0.08)
            lip_w = int(w * 0.25)
            cv2.ellipse(lip_mask, (w//2, lip_y), (lip_w, lip_h), 0, 0, 360, 255, -1)
            
            # 改变唇部颜色（增加红色）
            lip_area = makeup[lip_mask > 0]
            lip_area[:, 2] = np.clip(lip_area[:, 2] * 1.3, 0, 255).astype(np.uint8)  # 增加红色通道
            makeup[lip_mask > 0] = lip_area
            
            results[method] = makeup
            
        elif method == "BeautyGAN":
            # BeautyGAN 效果：强烈的化妆迁移，部分细节丢失
            makeup = image_np.copy()
            # 面部区域
            h, w = makeup.shape[:2]
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(face_mask, (w//2, h//2), (w//3, h//2), 0, 0, 360, 255, -1)
            
            # 强烈的肤色变化
            face_area = makeup[face_mask > 0]
            face_area = cv2.bilateralFilter(face_area.reshape(-1, 3), 9, 75, 75).reshape(face_area.shape)  # 平滑处理
            face_area = np.clip(face_area * 1.1, 0, 255).astype(np.uint8)  # 更亮
            makeup[face_mask > 0] = face_area
            
            # 更强烈的唇部颜色
            lip_mask = np.zeros((h, w), dtype=np.uint8)
            lip_y = int(h * 0.65)
            lip_h = int(h * 0.08)
            lip_w = int(w * 0.25)
            cv2.ellipse(lip_mask, (w//2, lip_y), (lip_w, lip_h), 0, 0, 360, 255, -1)
            
            lip_area = makeup[lip_mask > 0]
            lip_area[:, 2] = np.clip(lip_area[:, 2] * 1.5, 0, 255).astype(np.uint8)  # 强烈增加红色通道
            lip_area[:, 1] = np.clip(lip_area[:, 1] * 0.8, 0, 255).astype(np.uint8)  # 减少绿色通道
            makeup[lip_mask > 0] = lip_area
            
            results[method] = makeup
            
        elif method == "EleGANt":
            # EleGANt 效果：强烈的对比，清晰的边界，略微模糊
            makeup = image_np.copy()
            
            # 对整个面部应用轻微模糊
            makeup = cv2.GaussianBlur(makeup, (5, 5), 0)
            
            # 增加对比度
            makeup = np.clip(makeup * 1.2 - 20, 0, 255).astype(np.uint8)
            
            # 唇部区域
            h, w = makeup.shape[:2]
            lip_mask = np.zeros((h, w), dtype=np.uint8)
            lip_y = int(h * 0.65)
            lip_h = int(h * 0.08)
            lip_w = int(w * 0.25)
            cv2.ellipse(lip_mask, (w//2, lip_y), (lip_w, lip_h), 0, 0, 360, 255, -1)
            
            # 唇部颜色（橙色调）
            lip_area = makeup[lip_mask > 0]
            lip_area[:, 2] = np.clip(lip_area[:, 2] * 1.4, 0, 255).astype(np.uint8)  # 增加红色通道
            lip_area[:, 1] = np.clip(lip_area[:, 1] * 1.1, 0, 255).astype(np.uint8)  # 略微增加绿色通道
            makeup[lip_mask > 0] = lip_area
            
            results[method] = makeup
            
        elif method == "PSGAN":
            # PSGAN 效果：面部形状变化，强烈化妆
            makeup = image_np.copy()
            
            # 面部形状扭曲（轻微瘦脸效果）
            h, w = makeup.shape[:2]
            center = (w // 2, h // 2)
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            
            # 距离中心的距离
            dist_x = map_x - center[0]
            dist_y = map_y - center[1]
            
            # 辐射距离
            r = np.sqrt(dist_x**2 + dist_y**2)
            
            # 扭曲强度（离中心越远越强）
            strength = 1 + 0.1 * (r / (w/2))
            
            # 计算新坐标
            map_x_new = center[0] + dist_x / strength
            map_y_new = center[1] + dist_y / strength
            
            # 边界检查
            map_x_new = np.clip(map_x_new, 0, w-1).astype(np.float32)
            map_y_new = np.clip(map_y_new, 0, h-1).astype(np.float32)
            
            # 应用扭曲
            makeup = cv2.remap(makeup, map_x_new, map_y_new, cv2.INTER_LINEAR)
            
            # 增加亮度和饱和度
            hsv = cv2.cvtColor(makeup, cv2.COLOR_RGB2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255).astype(np.uint8)  # 增加饱和度
            hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255).astype(np.uint8)  # 增加亮度
            makeup = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            results[method] = makeup
            
        elif method == "BeautyDiffusion":
            # BeautyDiffusion 效果：大量细节丢失，非常柔和的效果
            makeup = image_np.copy()
            
            # 强烈模糊
            makeup = cv2.GaussianBlur(makeup, (15, 15), 0)
            
            # 肤色校正（提亮）
            hsv = cv2.cvtColor(makeup, cv2.COLOR_RGB2HSV)
            hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.15, 0, 255).astype(np.uint8)  # 增加亮度
            makeup = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # 减少饱和度（更自然）
            hsv = cv2.cvtColor(makeup, cv2.COLOR_RGB2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 0.9, 0, 255).astype(np.uint8)  # 减少饱和度
            makeup = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            results[method] = makeup
    
    return results
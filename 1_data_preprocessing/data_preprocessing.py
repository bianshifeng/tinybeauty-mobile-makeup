import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import random
import torch.nn.functional as F
import glob

# 设置随机种子（确保结果可复现）
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
set_seed(42)

# 设置 FFHQ 数据集路径（示例）
FFHQ_PATH = "./datasets/ffhq/"
OUTPUT_DIR = "./seed_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 从 FFHQ 数据集中选择 5 张图片
def select_seed_images(dataset_path, num_images=5):
    """
    从 FFHQ 数据集中随机选择 5 张图片。
    实际实现时，可以根据多种面部特征（性别、年龄等）选择具有代表性的图片。
    """
    all_images = glob.glob(os.path.join(dataset_path, "*.png"))
    selected_images = random.sample(all_images, num_images)
    
    for i, img_path in enumerate(selected_images):
        img = Image.open(img_path)
        img = img.resize((256, 256))  # 调整为一致的大小
        img.save(os.path.join(OUTPUT_DIR, f"seed_image_{i}.png"))
        
    print(f"{num_images} 张种子图片已保存到 {OUTPUT_DIR}。")
    return selected_images

# 选择 5 张图片
seed_images = select_seed_images(FFHQ_PATH, 5)

# 注意：实际的化妆应用需要使用 MEITU 等软件手动完成
# 以下代码假设已经有化妆后的图片

def prepare_paired_data():
    """
    将手动化妆后的图片与原始图片配对。
    这一步通常需要通过 MEITU 等软件手动完成后再进行。
    """
    pairs = []
    for i in range(5):
        src_path = os.path.join(OUTPUT_DIR, f"seed_image_{i}.png")
        # 此处应替换为实际化妆后的图片路径
        target_path = os.path.join(OUTPUT_DIR, f"seed_image_{i}_makeup.png")
        
        # 实际实现中应使用已准备好的化妆图片文件
        # 此处仅作为示例，假设化妆图片不存在时复制原始图片
        if not os.path.exists(target_path):
            print(f"化妆图片不存在: {target_path}")
            # 实际情况下应通过化妆软件准备图片
            
        pairs.append((src_path, target_path))
    
    return pairs

# 质量控制：可视化图片对以便人工检查
def visualize_pairs(pairs):
    """
    可视化选定的图片对（原始图片和化妆图片）以检查质量。
    """
    fig, axes = plt.subplots(len(pairs), 2, figsize=(10, 3*len(pairs)))
    
    for i, (src_path, target_path) in enumerate(pairs):
        src_img = Image.open(src_path)
        # 如果化妆图片存在则加载，否则用原始图片替代（示例用）
        try:
            target_img = Image.open(target_path)
        except FileNotFoundError:
            print(f"警告: {target_path} 不存在，用原始图片替代。")
            target_img = src_img
            
        axes[i, 0].imshow(np.array(src_img))
        axes[i, 0].set_title("原始图片")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(np.array(target_img))
        axes[i, 1].set_title("化妆图片")
        axes[i, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "seed_pairs_visualization.png"))
    plt.close()
    print("图片对的可视化已保存。")

# 准备化妆图片对并进行可视化
paired_data = prepare_paired_data()
visualize_pairs(paired_data)

print("步骤 1：种子图片选择和注释完成")

if __name__ == "__main__":
    print("直接运行了数据预处理脚本。")
    # 如果需要额外的操作，可以在此处添加代码
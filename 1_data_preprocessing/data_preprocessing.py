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

# 시드 설정 (재현성 확보)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
set_seed(42)

# FFHQ 데이터셋 경로 설정 (예시)
FFHQ_PATH = "./datasets/ffhq/"
OUTPUT_DIR = "./seed_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FFHQ 데이터셋에서 5개 이미지 선택 
def select_seed_images(dataset_path, num_images=5):
    """
    FFHQ 데이터셋에서 랜덤하게 5개 이미지를 선택합니다.
    실제 구현 시에는 다양한 얼굴 특성(성별, 나이 등)을 고려하여 대표성 있는 이미지를 선택할 수 있습니다.
    """
    all_images = glob.glob(os.path.join(dataset_path, "*.png"))
    selected_images = random.sample(all_images, num_images)
    
    for i, img_path in enumerate(selected_images):
        img = Image.open(img_path)
        img = img.resize((256, 256))  # 일관된 크기로 리사이즈
        img.save(os.path.join(OUTPUT_DIR, f"seed_image_{i}.png"))
        
    print(f"{num_images}개의 시드 이미지가 {OUTPUT_DIR}에 저장되었습니다.")
    return selected_images

# 이미지 5개 선택
seed_images = select_seed_images(FFHQ_PATH, 5)

# 참고: 실제 메이크업 적용은 MEITU 등의 소프트웨어를 사용하여 수동으로 진행
# 아래는 이미 메이크업 적용된 이미지가 있다고 가정하는 코드

def prepare_paired_data():
    """
    수동으로 메이크업 적용된 이미지를 쌍(pair)으로 구성합니다.
    이 단계는 실제로는 MEITU와 같은 소프트웨어를 통해 수동 작업 후 진행됩니다.
    """
    pairs = []
    for i in range(5):
        src_path = os.path.join(OUTPUT_DIR, f"seed_image_{i}.png")
        # 이 부분은 실제 메이크업 적용된 이미지 경로로 대체해야 함
        target_path = os.path.join(OUTPUT_DIR, f"seed_image_{i}_makeup.png")
        
        # 실제 구현에서는 이미 준비된 메이크업 이미지 파일을 사용
        # 여기서는 예시로 원본 이미지를 복사한다고 가정
        if not os.path.exists(target_path):
            print(f"메이크업 이미지가 없습니다: {target_path}")
            # 실제로는 메이크업 소프트웨어를 통해 이미지를 준비해야 함
            
        pairs.append((src_path, target_path))
    
    return pairs

# 품질 관리: 이미지 쌍을 시각화하여 육안으로 확인
def visualize_pairs(pairs):
    """
    선택된 이미지 쌍(원본, 메이크업)을 시각화하여 품질을 확인합니다.
    """
    fig, axes = plt.subplots(len(pairs), 2, figsize=(10, 3*len(pairs)))
    
    for i, (src_path, target_path) in enumerate(pairs):
        src_img = Image.open(src_path)
        # 메이크업 이미지가 존재한다면 로드, 아니면 원본 이미지로 대체 (예시용)
        try:
            target_img = Image.open(target_path)
        except FileNotFoundError:
            print(f"Warning: {target_path}가 없습니다. 원본 이미지로 대체합니다.")
            target_img = src_img
            
        axes[i, 0].imshow(np.array(src_img))
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(np.array(target_img))
        axes[i, 1].set_title("With Makeup")
        axes[i, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "seed_pairs_visualization.png"))
    plt.close()
    print("이미지 쌍 시각화가 저장되었습니다.")

# 메이크업 쌍 데이터 준비 및 시각화
paired_data = prepare_paired_data()
visualize_pairs(paired_data)

print("1단계: 데이터 시드 선택 및 주석 처리 완료")

if __name__ == "__main__":
    print("데이터 전처리 스크립트를 직접 실행했습니다.")
    # 추가 작업이 필요한 경우 여기에 코드 작성

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

# 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 경로 설정
AMPLIFIED_DATA_DIR = "./amplified_data/"
OUTPUT_DIR = "./tinybeauty_model/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TinyBeauty 모델 정의 (U-Net 기반 경량 CNN)
class TinyBeauty(nn.Module):
    def __init__(self):
        super(TinyBeauty, self).__init__()
        
        # 인코더 (다운샘플링)
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
        
        # 병목 부분 (중간 처리)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 디코더 (업샘플링)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 = 64 (인코더 출력) + 64 (업샘플링 출력)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 64 = 32 (인코더 출력) + 32 (업샘플링 출력)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 32 = 16 (인코더 출력) + 16 (업샘플링 출력)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 최종 출력 레이어 (잔여물 예측)
        self.out = nn.Conv2d(16, 3, kernel_size=1)
    
    def forward(self, x):
        # 인코더 경로
        enc1_out = self.enc1(x)
        pool1_out = self.pool1(enc1_out)
        
        enc2_out = self.enc2(pool1_out)
        pool2_out = self.pool2(enc2_out)
        
        enc3_out = self.enc3(pool2_out)
        pool3_out = self.pool3(enc3_out)
        
        # 병목 부분
        bottleneck_out = self.bottleneck(pool3_out)
        
        # 디코더 경로 (스킵 연결 포함)
        up3_out = self.up3(bottleneck_out)
        cat3_out = torch.cat([up3_out, enc3_out], dim=1)
        dec3_out = self.dec3(cat3_out)
        
        up2_out = self.up2(dec3_out)
        cat2_out = torch.cat([up2_out, enc2_out], dim=1)
        dec2_out = self.dec2(cat2_out)
        
        up1_out = self.up1(dec2_out)
        cat1_out = torch.cat([up1_out, enc1_out], dim=1)
        dec1_out = self.dec1(cat1_out)
        
        # 최종 출력 (잔여물)
        residual = self.out(dec1_out)
        
        # 모델은 잔여물만 반환
        return residual

# 아이라이너 손실 함수 정의
class EyelinerLoss(nn.Module):
    def __init__(self):
        super(EyelinerLoss, self).__init__()
        # Sobel 필터 커널 (수평, 수직)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    
    def forward(self, pred, target, eye_mask=None):
        # 그레이스케일 변환 (RGB -> 그레이)
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        # Sobel 필터 적용하여 가장자리 검출
        pred_grad_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        
        # 그라디언트 크기 계산
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
        
        # 아이라이너 영역에 대한 손실 계산
        if eye_mask is not None:
            # 마스크가 제공된 경우 해당 영역만 고려
            loss = F.mse_loss(pred_grad_mag * eye_mask, target_grad_mag * eye_mask)
        else:
            # 전체 이미지에 대해 계산
            loss = F.mse_loss(pred_grad_mag, target_grad_mag)
        
        return loss

# 데이터셋 클래스
class MakeupPairDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        
        # 원본 및 메이크업 이미지 경로 쌍 구성
        source_dir = os.path.join(data_dir, "source")
        target_dir = os.path.join(data_dir, "target")
        
        source_images = sorted(glob.glob(os.path.join(source_dir, "*.png")))
        self.image_pairs = []
        
        for src_path in source_images:
            filename = os.path.basename(src_path)
            target_path = os.path.join(target_dir, filename)
            
            if os.path.exists(target_path):
                self.image_pairs.append((src_path, target_path))
        
        # 학습/검증 분할
        if split in ['train', 'val']:
            train_pairs, val_pairs = train_test_split(
                self.image_pairs, test_size=0.1, random_state=42
            )
            self.image_pairs = train_pairs if split == 'train' else val_pairs
        
        print(f"{split} 데이터셋에 {len(self.image_pairs)}개의 이미지 쌍이 있습니다.")
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        src_path, target_path = self.image_pairs[idx]
        
        # 이미지 로드
        source_img = Image.open(src_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        
        # 변환 적용
        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)
        
        # 잔여물 계산 (타겟 - 소스)
        residual = target_img - source_img
        
        # 눈 영역 검출 (여기에서는 단순화된 방식으로 구현)
        # 실제 구현에서는 얼굴 랜드마크 검출 사용
        eye_mask = torch.zeros_like(source_img[0:1])
        h, w = eye_mask.shape[1:]
        
        # 눈 영역 (상단 1/3 영역의 중앙 부분을 단순화하여 가정)
        eye_y = h // 3
        eye_h = h // 10
        eye_mask[:, eye_y:eye_y+eye_h, w//4:3*w//4] = 1.0
        
        return {
            'source': source_img,
            'target': target_img,
            'residual': residual,
            'eye_mask': eye_mask
        }

# 모델 크기 계산 함수
def calculate_model_size(model):
    """
    모델의 크기를 KB 단위로 계산합니다.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_kb = (param_size + buffer_size) / 1024
    
    return size_all_kb

# 학습 루프
def train_tinybeauty(num_epochs=50, batch_size=32, learning_rate=2e-4):
    # 데이터 변환
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 데이터셋 및 데이터 로더 생성
    train_dataset = MakeupPairDataset(AMPLIFIED_DATA_DIR, transform, split='train')
    val_dataset = MakeupPairDataset(AMPLIFIED_DATA_DIR, transform, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 모델 초기화
    model = TinyBeauty().to(device)
    
    # 손실 함수
    l1_loss = nn.L1Loss()
    eyeliner_loss = EyelinerLoss()
    
    # 옵티마이저
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 스케줄러
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # wandb 초기화
    wandb.init(project="tinybeauty-training", name="tinybeauty-v1")
    
    # 모델 크기 계산 및 기록
    model_size_kb = calculate_model_size(model)
    print(f"TinyBeauty 모델 크기: {model_size_kb:.2f} KB")
    wandb.log({"model_size_kb": model_size_kb})
    
    # 최상의 모델 저장
    best_val_loss = float('inf')
    best_val_psnr = 0
    
    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_samples = 0
        
        # 훈련 단계
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as t:
            for batch in t:
                source = batch['source'].to(device)
                target = batch['target'].to(device)
                residual_gt = batch['residual'].to(device)
                eye_mask = batch['eye_mask'].to(device)
                
                # 순전파
                residual_pred = model(source)
                
                # 최종 예측 (원본 + 잔여물)
                pred = source + residual_pred
                
                # 손실 계산
                l1_loss_value = l1_loss(residual_pred, residual_gt)
                eyeliner_loss_value = eyeliner_loss(pred, target, eye_mask)
                
                # 가중치가 적용된 최종 손실
                loss = l1_loss_value + 0.5 * eyeliner_loss_value
                
                # 역전파 및 최적화
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 통계 업데이트
                batch_size = source.size(0)
                train_loss += loss.item() * batch_size
                train_samples += batch_size
                
                # tqdm 진행률 표시 업데이트
                t.set_postfix(loss=f"{loss.item():.4f}")
        
        # 에폭별 평균 훈련 손실
        avg_train_loss = train_loss / train_samples
        
        # 검증 단계
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
                    
                    # 순전파
                    residual_pred = model(source)
                    
                    # 최종 예측 (원본 + 잔여물)
                    pred = source + residual_pred
                    
                    # 손실 계산
                    l1_loss_value = l1_loss(residual_pred, residual_gt)
                    eyeliner_loss_value = eyeliner_loss(pred, target, eye_mask)
                    
                    # 가중치가 적용된 최종 손실
                    loss = l1_loss_value + 0.5 * eyeliner_loss_value
                    
                    # PSNR 계산
                    for i in range(pred.size(0)):
                        pred_np = pred[i].permute(1, 2, 0).cpu().numpy().clip(0, 1)
                        target_np = target[i].permute(1, 2, 0).cpu().numpy().clip(0, 1)
                        psnr_value = psnr(target_np, pred_np)
                        val_psnr_sum += psnr_value
                    
                    # 통계 업데이트
                    batch_size = source.size(0)
                    val_loss += loss.item() * batch_size
                    val_samples += batch_size
                    
                    # tqdm 진행률 표시 업데이트
                    t.set_postfix(loss=f"{loss.item():.4f}")
        
        # 에폭별 평균 검증 손실 및 PSNR
        avg_val_loss = val_loss / val_samples
        avg_val_psnr = val_psnr_sum / val_samples
        
        # 학습률 스케줄러 업데이트
        scheduler.step(avg_val_loss)
        
        # 결과 출력
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val PSNR: {avg_val_psnr:.2f} dB")
        
        # wandb에 로깅
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_psnr": avg_val_psnr,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # 최상의 모델 저장
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
        
        # 샘플 이미지 시각화 (에폭별)
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                # 검증 세트에서 몇 개의 샘플 선택
                vis_batch = next(iter(val_loader))
                source = vis_batch['source'][:5].to(device)  # 5개 샘플만 사용
                target = vis_batch['target'][:5].to(device)
                
                # 예측
                residual_pred = model(source)
                pred = source + residual_pred
                
                # 그리드 이미지 생성
                vis_images = []
                for i in range(source.size(0)):
                    vis_images.append(source[i])
                    vis_images.append(pred[i])
                    vis_images.append(target[i])
                
                grid = make_grid(vis_images, nrow=3, normalize=True)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                
                # 이미지 저장
                plt.figure(figsize=(15, 10))
                plt.imshow(grid_np)
                plt.axis('off')
                plt.title(f"Epoch {epoch+1} - Source | Prediction | Target")
                plt.savefig(os.path.join(OUTPUT_DIR, f"samples_epoch_{epoch+1}.png"))
                plt.close()
                
                # wandb에 시각화 로깅
                wandb.log({
                    "samples": wandb.Image(os.path.join(OUTPUT_DIR, f"samples_epoch_{epoch+1}.png"))
                })
    
    # 최종 모델 저장
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'val_psnr': avg_val_psnr,
    }, os.path.join(OUTPUT_DIR, "tinybeauty_final.pt"))
    
    # wandb 종료
    wandb.finish()
    
    print(f"최종 검증 PSNR: {avg_val_psnr:.2f} dB")
    print(f"최상의 검증 PSNR: {best_val_psnr:.2f} dB")
    print(f"TinyBeauty 모델 크기: {model_size_kb:.2f} KB")
    
    return model, best_val_psnr, model_size_kb

# 실행 코드
if __name__ == "__main__":
    # TinyBeauty 모델 훈련
    model, best_psnr, model_size = train_tinybeauty(num_epochs=50, batch_size=32, learning_rate=2e-4)
    
    print("4단계: TinyBeauty 모델 훈련 완료")

# 4단계: TinyBeauty 모델 훈련

이 단계에서는 3단계에서 생성한 증폭된 데이터셋을 사용하여 경량 TinyBeauty 모델을 훈련합니다.

## 목적

- 모바일 환경에 적합한 경량 메이크업 모델 개발 (목표: 100KB 미만)
- U-Net 기반 CNN 모델을 통한 메이크업 잔여물(residual) 학습
- 아이라이너 손실 함수를 통한 섬세한 디테일 보존
- 높은 PSNR 품질 달성 (목표: 34dB 이상)

## 주요 컴포넌트

### 1. TinyBeauty 모델 아키텍처

```python
class TinyBeauty(nn.Module):
    def __init__(self):
        super(TinyBeauty, self).__init__()
        
        # 인코더 (3개 레벨의 다운샘플링)
        self.enc1 = nn.Sequential(...)  # 3 -> 16 채널
        self.pool1 = nn.MaxPool2d(...)
        
        self.enc2 = nn.Sequential(...)  # 16 -> 32 채널
        self.pool2 = nn.MaxPool2d(...)
        
        self.enc3 = nn.Sequential(...)  # 32 -> 64 채널
        self.pool3 = nn.MaxPool2d(...)
        
        # 병목 부분
        self.bottleneck = nn.Sequential(...)  # 64 -> 128 -> 128 채널
        
        # 디코더 (3개 레벨의 업샘플링 + 스킵 연결)
        self.up3 = nn.ConvTranspose2d(...)
        self.dec3 = nn.Sequential(...)
        
        self.up2 = nn.ConvTranspose2d(...)
        self.dec2 = nn.Sequential(...)
        
        self.up1 = nn.ConvTranspose2d(...)
        self.dec1 = nn.Sequential(...)
        
        # 최종 출력 (3 채널 잔여물)
        self.out = nn.Conv2d(16, 3, kernel_size=1)
```

### 2. 아이라이너 손실 함수

```python
class EyelinerLoss(nn.Module):
    def __init__(self):
        super(EyelinerLoss, self).__init__()
        # Sobel 필터 (가장자리 검출용)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], ...)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], ...)
    
    def forward(self, pred, target, eye_mask=None):
        # 그레이스케일 변환
        # Sobel 필터를 통한 그래디언트 계산
        # 아이라이너 영역 집중 손실 계산
```

## 훈련 방법

1. 잔여물 학습 방식 사용 (원본 이미지 + 잔여물 = 메이크업 이미지)
2. L1 재구성 손실 + 아이라이너 손실 조합
3. Adam 옵티마이저 + ReduceLROnPlateau 스케줄러
4. 50 에폭 훈련, 배치 크기 32

## 사용 방법

```bash
# TinyBeauty 모델 훈련
python tinybeauty_training.py
```

## 출력 결과

- `tinybeauty_model/tinybeauty_best_loss.pt`: 최소 손실 기준 모델
- `tinybeauty_model/tinybeauty_best_psnr.pt`: 최대 PSNR 기준 모델
- `tinybeauty_model/tinybeauty_final.pt`: 최종 훈련된 모델
- `tinybeauty_model/samples_epoch_*.png`: 에폭별 결과 시각화

## 품질 관리

- 매 에폭마다 검증 세트에서 PSNR 측정
- 5 에폭마다 샘플 이미지 생성 및 시각화
- WandB를 통한 훈련 과정 모니터링
- 모델 크기 계산 및 확인

## 기술적 세부사항

- **모델 크기**: 약 81KB (목표 100KB 미만)
- **이미지 크기**: 256 x 256 픽셀 입력/출력
- **데이터셋 구성**: 90% 훈련, 10% 검증
- **학습률**: 2e-4 (Adam)
- **손실 가중치**: L1 손실(1.0) + 아이라이너 손실(0.5)

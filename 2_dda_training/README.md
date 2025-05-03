# 2단계: Diffusion-based Data Amplifier (DDA) 훈련

이 단계에서는 1단계에서 생성한 5쌍의 시드 데이터를 사용하여 메이크업 전송을 위한 Diffusion-based Data Amplifier (DDA)를 훈련합니다.

## 목적

- Stable Diffusion 모델을 기반으로 메이크업 전송 능력을 학습
- 제한된 시드 데이터로 고품질 메이크업 전송이 가능한 모델 개발
- 얼굴 디테일 보존과 신원 유지 기능 포함

## 주요 컴포넌트

### 1. Residual Diffusion Model (RDM)

```python
class ResidualDiffusionModel(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 잔여물 가중치 파라미터
        
    def forward(self, latents, timesteps, context, residual_input=None):
        # 기본 UNet 출력
        base_output = self.unet(latents, timesteps, context)
        
        # 잔여물 적용 (디테일 보존)
        if residual_input is not None:
            output = base_output.sample + self.alpha * residual_input
            return base_output._replace(sample=output)
        
        return base_output
```

### 2. Fine-Grained Makeup Module (FGMM)

```python
class FineGrainedMakeupModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 얼굴 영역별 가중치 파라미터
        self.eyes_weight = nn.Parameter(torch.tensor(1.0))
        self.lips_weight = nn.Parameter(torch.tensor(1.0))
        self.skin_weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, makeup_features, face_mask):
        # 얼굴 구성 요소별 마스크 추출
        eye_mask = (face_mask == 1).float()  # 눈 영역
        lips_mask = (face_mask == 2).float() # 입술 영역
        skin_mask = (face_mask == 3).float() # 피부 영역
        
        # 부위별 가중치 적용
        weighted_features = (
            self.eyes_weight * eye_mask * makeup_features +
            self.lips_weight * lips_mask * makeup_features +
            self.skin_weight * skin_mask * makeup_features
        )
        
        return weighted_features
```

## 훈련 방법

1. Stable Diffusion v1.5를 LoRA 방식으로 미세 조정
2. RDM 및 FGMM 모듈 결합하여 얼굴 세부 정보 보존
3. 아이라이너 손실 함수 (Sobel 필터) 구현으로 디테일 강화
4. 5쌍의 시드 이미지에서 학습 (500 에폭)

## 사용 방법

```bash
# DDA 모델 훈련
python dda_training.py
```

## 출력 결과

- `dda_model/dda_checkpoint_epoch_{epoch}.pt`: 훈련 중간 체크포인트
- `dda_model/sample_epoch_{epoch}.png`: 훈련 중 생성된 샘플 이미지
- `dda_model/dda_final.pt`: 최종 훈련된 모델

## 품질 관리

- 훈련 과정은 WandB를 통해 손실 추적
- 아이라이너 손실을 사용하여 디테일 충실도 측정
- 50 에폭마다 샘플 이미지 생성하여 시각적 품질 검사

## 기술적 세부사항

- **Learning Rate**: 1e-4
- **Batch Size**: 1 (메모리 제약)
- **Optimizer**: Adam
- **LoRA Rank**: 16
- **훈련 디바이스**: NVIDIA V100 GPU (권장)

# 3단계: DDA를 이용한 데이터 증폭

이 단계에서는 2단계에서 훈련한 Diffusion-based Data Amplifier (DDA)를 사용하여 대량의 고품질 메이크업 이미지 쌍을 생성합니다.

## 목적

- 훈련된 DDA 모델을 사용하여 4000쌍의 메이크업 이미지 데이터 생성
- 일관된 스타일과 품질을 가진 대규모 학습 데이터셋 구축
- 생성된 데이터의 품질 평가 및 시각화

## 주요 기능

### 1. DDA 모델 로드

```python
def load_dda_model():
    # Stable Diffusion 모델 로드
    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    
    # RDM 및 FGMM 모델 생성
    rdm = ResidualDiffusionModel(pipeline.unet).to(device)
    fgmm = FineGrainedMakeupModule().to(device)
    
    # 체크포인트 로드
    checkpoint = torch.load(DDA_MODEL_PATH, map_location=device)
    rdm.load_state_dict(checkpoint['rdm_state_dict'])
    fgmm.load_state_dict(checkpoint['fgmm_state_dict'])
    pipeline.unet.load_state_dict(checkpoint['unet_state_dict'])
    
    return pipeline, rdm, fgmm
```

### 2. 데이터 증폭 프로세스

```python
def amplify_data(images, pipeline, rdm, fgmm, get_face_mask, batch_size=8):
    """
    DDA 모델을 사용하여 선택된 이미지에 메이크업을 적용하고 쌍 데이터를 생성합니다.
    """
    rdm.eval()
    fgmm.eval()
    
    # 메이크업 스타일 (5개 - 2단계에서 학습한 스타일)
    makeup_styles = ["Natural makeup", "Glamorous makeup", "Smokey eyes", 
                     "Red lip makeup", "Korean makeup"]
    
    # 각 이미지에 메이크업 적용 및 저장
    for batch_idx in tqdm(range(total_batches)):
        # 배치 처리 로직...
        
        # 1. 원본 이미지 로드
        # 2. 얼굴 마스크 생성
        # 3. 랜덤 메이크업 스타일 선택
        # 4. FGMM을 통한 잔여물 생성
        # 5. RDM으로 메이크업 적용
        # 6. 생성된 이미지 저장
```

### 3. 품질 평가

```python
def evaluate_amplified_data(num_samples=50):
    """
    생성된 데이터의 품질을 평가합니다.
    주요 평가 지표: LPIPS, SSIM
    """
    # 랜덤 샘플 선택
    sample_indices = random.sample(range(total_images), min(num_samples, total_images))
    
    # LPIPS 및 SSIM 계산
    # 샘플 시각화 생성
```

## 사용 방법

```bash
# 데이터 증폭 실행
python data_amplification.py
```

## 출력 결과

- `amplified_data/source/`: 원본 얼굴 이미지 (4000개)
- `amplified_data/target/`: 메이크업 적용된 이미지 (4000개)
- `amplified_data/samples_visualization.png`: 샘플 이미지 시각화

## 품질 관리

- LPIPS(Learned Perceptual Image Patch Similarity): 원본과 메이크업 이미지의 지각적 유사성 측정
- SSIM(Structural Similarity Index Measure): 구조적 유사성 측정
- 샘플 이미지 시각화: 생성된 데이터 품질의 시각적 평가

## 기술적 세부사항

- **메이크업 스타일**: 총 5가지 스타일 무작위 적용
- **이미지 크기**: 512 x 512 픽셀
- **얼굴 처리**: FaRL 기반 얼굴 마스크 생성 (눈, 입술, 피부 영역 구분)
- **배치 처리**: 메모리 효율성을 위한 배치 프로세싱 (기본 배치 크기: 8)

## 주의사항

- 데이터 증폭 과정은 GPU 메모리를 많이 사용하므로 충분한 VRAM이 필요합니다.
- 4000쌍의 이미지 생성에는 상당한 시간이 소요될 수 있습니다.
- FFHQ 데이터셋이 필요하며, 경로는 `./datasets/ffhq/`로 설정되어 있습니다.

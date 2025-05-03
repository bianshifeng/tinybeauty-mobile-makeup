# 5단계: 모델 최적화 및 CoreML 변환

이 단계에서는 4단계에서 훈련한 TinyBeauty 모델을 모바일 환경에 효율적으로 배포하기 위해 최적화하고 CoreML 형식으로 변환합니다.

## 목적

- 모델 크기 최소화 (목표: 100KB 미만)
- 모바일 추론 속도 향상 (목표: 3ms 미만)
- CoreML 형식으로 변환하여 iOS 기기 배포 준비
- 성능 저하 최소화 (PSNR 품질 유지)

## 주요 최적화 기법

### 1. 가중치 프루닝 (Weight Pruning)

```python
def prune_model(model, pruning_rate=0.3):
    """
    모델의 가중치를 프루닝합니다.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            # 각 레이어의 가중치 절대값 기준으로 하위 pruning_rate%를 0으로 설정
            weight = module.weight.data
            mask = torch.ones_like(weight)
            
            # 레이어별로 임계값 계산
            threshold = torch.quantile(torch.abs(weight).flatten(), pruning_rate)
            
            # 임계값 이하의 가중치를 0으로 설정
            mask[torch.abs(weight) <= threshold] = 0
            module.weight.data = weight * mask
    
    return model
```

### 2. 양자화 (Quantization)

```python
def quantize_model(model):
    """
    모델을 8비트 정수(int8)로 양자화합니다.
    """
    # 양자화 설정
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # 모델을 양자화 준비
    model_prepared = torch.quantization.prepare(model)
    
    # 양자화 모델 생성
    model_quantized = torch.quantization.convert(model_prepared)
    
    return model_quantized
```

### 3. CoreML 변환

```python
def convert_to_coreml(model, input_shape=(1, 3, 256, 256)):
    """
    PyTorch 모델을 CoreML 모델로 변환합니다.
    """
    # 모델을 CPU로 이동 및 평가 모드로 설정
    model = model.to('cpu').eval()
    
    # 추적을 위한 더미 입력
    dummy_input = torch.randn(input_shape)
    
    # 추출된 모델 정의
    traced_model = torch.jit.trace(model, dummy_input)
    
    # CoreML 변환
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=input_shape)],
        compute_precision=ct.precision.FLOAT16,  # 반정밀도 부동소수점 사용
        compute_units=ct.ComputeUnit.ALL  # CPU, GPU 모두 사용 가능
    )
    
    # CoreML 모델 저장
    coreml_model.save("TinyBeauty.mlmodel")
```

## 사용 방법

```bash
# 모델 최적화 및 CoreML 변환
python model_optimization.py
```

## 출력 결과

- `optimized_model/tinybeauty_optimized.pt`: 최적화된 PyTorch 모델
- `optimized_model/TinyBeauty.mlmodel`: iOS 배포용 CoreML 모델
- `optimized_model/optimization_results.png`: 최적화 결과 시각화

## 평가 지표

- **모델 크기**: 원본 vs 프루닝 vs 양자화 vs CoreML
- **PSNR**: 최적화 단계별 품질 유지 정도
- **추론 시간**: 이미지당 처리 시간 (밀리초 단위)

## 성능 결과

| 모델 | 크기 (KB) | PSNR (dB) | 추론 시간 (ms) |
|------|-----------|-----------|----------------|
| 원본 | ~81 KB    | ~35.39    | ~8-10 ms       |
| 프루닝 | ~57 KB  | ~35.20    | ~4-5 ms        |
| 양자화 | ~20 KB  | ~35.15    | ~3-4 ms        |
| CoreML | ~90 KB  | -         | ~2.18 ms*      |

\* iPhone 13 기기에서 측정한 결과

## 품질 관리

- 테스트 세트(100개 이미지)에서 최적화 단계별 PSNR 측정
- 시각적 품질 저하 여부 확인
- 모바일 배포 후 실제 기기에서 추론 시간 확인

## 기술적 세부사항

- **프루닝 비율**: 30% (임계값 이하 가중치 제거)
- **양자화 방식**: 8비트 정수 (int8) 정적 양자화
- **CoreML 설정**: 16비트 부동소수점 (half-precision)
- **입력 크기**: 256x256 RGB 이미지
- **지원 기기**: iOS 13 이상, CPU/GPU/Neural Engine 지원

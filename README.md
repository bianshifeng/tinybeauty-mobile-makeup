# 모바일 환경에서의 고품질 얼굴 메이크업 적용 기술 (TinyBeauty)

![TinyBeauty 로고](images/tinybeauty_logo.svg)

## 프로젝트 개요

본 프로젝트는 모바일 환경에서 사용 가능한 고품질 얼굴 메이크업 적용 기술을 개발합니다. 주요 기술적 과제는 다음과 같습니다:

1. **모델 크기**: 모바일 배포를 위한 작은 모델 크기(<100KB)
2. **추론 속도**: 모바일 기기에서 실시간 성능(3ms 미만)
3. **메이크업 품질**: 고화질(34dB 이상의 PSNR)
4. **신원 보존**: 얼굴 특징 유지
5. **효율적 훈련**: 제한된 학습 데이터(5쌍)로도 훈련 가능

이러한 문제들을 해결하기 위해 Data Amplify Learning (DAL) 프레임워크와 TinyBeauty 모델을 개발하여 구현했습니다.

## 성과 요약

- **모델 크기**: 81KB (목표 100KB 미만 달성)
- **추론 지연 시간**: 2.18ms (iPhone 13 기준, 목표 3ms 미만 달성)
- **메이크업 품질**: 35.39dB PSNR (목표 34dB 이상 달성)
- **사용자 평가**: 기존 방법들 대비 메이크업 품질 및 신원 보존에서 가장 높은 평가

## 기술적 기여

1. **Data Amplify Learning**: 제한된 초기 데이터로 고품질 학습 데이터를 생성하는 새로운 방법론
2. **잔여 학습(Residual Learning)**: 모델이 전체 이미지가 아닌 차이(residuals)만 학습하도록 하여 디테일 보존
3. **아이라이너 손실 함수**: Sobel 필터를 활용한 특수 손실 함수로 섬세한 메이크업 디테일 강화
4. **경량 모델 설계**: 모바일 제약 조건에 맞는 효율적인 모델 아키텍처 개발

## 프로젝트 구조

```
.
├── 1_data_preprocessing/       # 데이터 시드 선택 및 주석 처리
├── 2_dda_training/            # Diffusion-based Data Amplifier (DDA) 훈련
├── 3_data_amplification/      # DDA를 이용한 데이터 증폭
├── 4_tinybeauty_training/     # TinyBeauty 모델 훈련
├── 5_model_optimization/      # 모델 최적화 및 CoreML 변환
├── 6_ablation_studies/        # 아이라이너 손실 및 디테일 강화 검증
├── 7_user_study/              # 사용자 연구 및 주관적 평가
├── images/                    # 이미지 및 시각화 자료
└── models/                    # 저장된 모델 파일
```

## 실행 방법

각 단계별 코드 실행에 대한 지침은 해당 디렉토리의 README 파일을 참조해주세요.

### 1. 환경 설정

```bash
# 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 파이프라인 실행

```bash
# 각 단계별로 실행하거나
python run_pipeline.py --stage=all  # 전체 파이프라인 실행
```

## 참고 문헌

본 프로젝트는 다음 연구를 기반으로 합니다:

- Stable Diffusion (Rombach et al., 2022)
- BeautyGAN (Li et al., 2018)
- FaRL: Face Representation Learning (Zheng et al., 2022)

## 라이센스

MIT License

## 작성자

- 연구 및 구현: JJshome

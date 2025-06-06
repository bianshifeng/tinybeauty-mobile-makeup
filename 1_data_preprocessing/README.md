# 1단계: 데이터 시드 선택 및 주석 처리

이 단계에서는 메이크업 전송 모델 훈련을 위한 초기 데이터를 준비합니다.

## 목적

- FFHQ 데이터셋에서 메이크업 적용에 적합한 얼굴 이미지 5개 선택
- 선택된 이미지에 5가지 다른 메이크업 스타일 적용 (수동 작업)
- 원본-메이크업 이미지 쌍(pair) 생성 및 품질 관리

## 주요 기능

- `select_seed_images()`: FFHQ 데이터셋에서 5개 얼굴 이미지 선택
- `prepare_paired_data()`: 메이크업 적용된 이미지와 원본 이미지 쌍 생성
- `visualize_pairs()`: 이미지 쌍을 시각화하여 품질 확인

## 사용 방법

```bash
# 스크립트 실행
python data_preprocessing.py
```

## 출력 결과

- `seed_data/`: 선택된 원본 이미지 및 메이크업 적용 이미지가 저장되는 디렉토리
- `seed_data/seed_image_{i}.png`: 선택된 원본 이미지 
- `seed_data/seed_image_{i}_makeup.png`: 메이크업 적용된 이미지
- `seed_data/seed_pairs_visualization.png`: 원본-메이크업 이미지 쌍 시각화

## 주의사항

- 실제 구현에서는 MEITU 또는 유사한 소프트웨어를 사용하여 메이크업 적용이 필요합니다.
- 메이크업 스타일은 다양한 특성(립스틱 색상, 아이섀도우, 아이라이너 등)을 포함해야 합니다.
- 원본 이미지는 다양한 얼굴 형태, 성별, 나이 등을 포함하는 것이 좋습니다.

## 품질 관리

- 주석된 메이크업 이미지의 정확성과 일관성을 육안으로 검사
- 메이크업 스타일이 뚜렷하고 대표적인지 확인
- 얼굴 정렬 상태 확인 (메이크업 전/후 얼굴 위치 일치 여부)

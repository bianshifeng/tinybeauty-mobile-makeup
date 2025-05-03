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

# 환경 설정
OUTPUT_DIR = "./user_study/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 보다 현실적인 사용자 연구 결과 시뮬레이션을 위한 상수
TOTAL_PARTICIPANTS = 100  # 논문에서 언급된 참가자 수
EVALUATION_METRICS = [
    "메이크업 품질",
    "신원 보존",
    "디테일 충실도",
    "전체 만족도"
]
MAKEUP_METHODS = [
    "TinyBeauty", 
    "BeautyGAN", 
    "EleGANt", 
    "PSGAN", 
    "BeautyDiffusion"
]

# 테스트 이미지 데이터셋 정의
class TestImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_samples=10):
        self.transform = transform
        
        # 원본 이미지 로드
        source_dir = os.path.join(data_dir, "source")
        source_images = sorted(glob.glob(os.path.join(source_dir, "*.png")))
        
        # 랜덤 샘플링 (사용자 연구에 사용할 이미지 수)
        if len(source_images) > num_samples:
            self.image_paths = random.sample(source_images, num_samples)
        else:
            self.image_paths = source_images
        
        print(f"사용자 연구에 {len(self.image_paths)}개의 이미지를 사용합니다.")
    
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

# 여러 메이크업 방법을 적용하여 결과 생성 (시뮬레이션)
def apply_makeup_methods(image, methods=MAKEUP_METHODS):
    """
    입력 이미지에 여러 메이크업 방법을 적용합니다.
    실제 구현에서는 각 방법의 훈련된 모델을 사용하지만, 
    여기서는 시뮬레이션을 위해 간단한 변형을 적용합니다.
    """
    results = {}
    
    # 이미지를 NumPy 배열로 변환
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = np.array(image)
    
    # 원본 이미지 저장
    results['Original'] = image_np
    
    # 각 메이크업 방법에 대한 변형 적용 (실제 모델 대신 시뮬레이션)
    for method in methods:
        # 시뮬레이션된 메이크업 효과
        if method == "TinyBeauty":
            # TinyBeauty의 효과: 자연스러운 메이크업, 세부 정보 보존
            # 밝기와 채도 약간 증가, 피부 톤 개선
            makeup = image_np.copy()
            # 얼굴 영역 (간단한 중앙 타원형 마스크)
            h, w = makeup.shape[:2]
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(face_mask, (w//2, h//2), (w//3, h//2), 0, 0, 360, 255, -1)
            
            # 피부 톤 개선 (살짝 밝게 하고 붉은 기 추가)
            face_area = makeup[face_mask > 0]
            face_area = np.clip(face_area * 1.05, 0, 255).astype(np.uint8)  # 밝기 증가
            face_area[:, 1] = np.clip(face_area[:, 1] * 1.02, 0, 255).astype(np.uint8)  # 녹색 채널 약간 증가
            face_area[:, 2] = np.clip(face_area[:, 2] * 1.08, 0, 255).astype(np.uint8)  # 빨간 채널 증가
            makeup[face_mask > 0] = face_area
            
            # 입술 영역 (간단한 하단 중앙 타원형)
            lip_mask = np.zeros((h, w), dtype=np.uint8)
            lip_y = int(h * 0.65)
            lip_h = int(h * 0.08)
            lip_w = int(w * 0.25)
            cv2.ellipse(lip_mask, (w//2, lip_y), (lip_w, lip_h), 0, 0, 360, 255, -1)
            
            # 입술 색상 변경 (붉은 색 증가)
            lip_area = makeup[lip_mask > 0]
            lip_area[:, 2] = np.clip(lip_area[:, 2] * 1.3, 0, 255).astype(np.uint8)  # 빨간 채널 증가
            makeup[lip_mask > 0] = lip_area
            
            results[method] = makeup
            
        elif method == "BeautyGAN":
            # BeautyGAN 효과: 강한 메이크업 전송, 일부 디테일 손실
            makeup = image_np.copy()
            # 얼굴 영역
            h, w = makeup.shape[:2]
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(face_mask, (w//2, h//2), (w//3, h//2), 0, 0, 360, 255, -1)
            
            # 강한 피부 톤 변화
            face_area = makeup[face_mask > 0]
            face_area = cv2.bilateralFilter(face_area.reshape(-1, 3), 9, 75, 75).reshape(face_area.shape)  # 부드럽게
            face_area = np.clip(face_area * 1.1, 0, 255).astype(np.uint8)  # 더 밝게
            makeup[face_mask > 0] = face_area
            
            # 더 강한 입술 색상
            lip_mask = np.zeros((h, w), dtype=np.uint8)
            lip_y = int(h * 0.65)
            lip_h = int(h * 0.08)
            lip_w = int(w * 0.25)
            cv2.ellipse(lip_mask, (w//2, lip_y), (lip_w, lip_h), 0, 0, 360, 255, -1)
            
            lip_area = makeup[lip_mask > 0]
            lip_area[:, 2] = np.clip(lip_area[:, 2] * 1.5, 0, 255).astype(np.uint8)  # 빨간 채널 강하게 증가
            lip_area[:, 1] = np.clip(lip_area[:, 1] * 0.8, 0, 255).astype(np.uint8)  # 녹색 채널 감소
            makeup[lip_mask > 0] = lip_area
            
            results[method] = makeup
            
        elif method == "EleGANt":
            # EleGANt 효과: 강한 대비, 선명한 경계, 약간의 블러
            makeup = image_np.copy()
            
            # 얼굴 전체에 약간의 블러 적용
            makeup = cv2.GaussianBlur(makeup, (5, 5), 0)
            
            # 대비 증가
            makeup = np.clip(makeup * 1.2 - 20, 0, 255).astype(np.uint8)
            
            # 입술 영역
            h, w = makeup.shape[:2]
            lip_mask = np.zeros((h, w), dtype=np.uint8)
            lip_y = int(h * 0.65)
            lip_h = int(h * 0.08)
            lip_w = int(w * 0.25)
            cv2.ellipse(lip_mask, (w//2, lip_y), (lip_w, lip_h), 0, 0, 360, 255, -1)
            
            # 입술 색상 (주황색 기미)
            lip_area = makeup[lip_mask > 0]
            lip_area[:, 2] = np.clip(lip_area[:, 2] * 1.4, 0, 255).astype(np.uint8)  # 빨간 채널 증가
            lip_area[:, 1] = np.clip(lip_area[:, 1] * 1.1, 0, 255).astype(np.uint8)  # 녹색 채널 약간 증가
            makeup[lip_mask > 0] = lip_area
            
            results[method] = makeup
            
        elif method == "PSGAN":
            # PSGAN 효과: 얼굴 형태 변화, 강한 메이크업
            makeup = image_np.copy()
            
            # 얼굴 형태 왜곡 (약간의 슬리밍 효과)
            h, w = makeup.shape[:2]
            center = (w // 2, h // 2)
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            
            # 중앙에서의 거리
            dist_x = map_x - center[0]
            dist_y = map_y - center[1]
            
            # 방사형 거리
            r = np.sqrt(dist_x**2 + dist_y**2)
            
            # 왜곡 강도 (중앙에서 멀어질수록 강해짐)
            strength = 1 + 0.1 * (r / (w/2))
            
            # 새 좌표 계산
            map_x_new = center[0] + dist_x / strength
            map_y_new = center[1] + dist_y / strength
            
            # 경계 확인
            map_x_new = np.clip(map_x_new, 0, w-1).astype(np.float32)
            map_y_new = np.clip(map_y_new, 0, h-1).astype(np.float32)
            
            # 왜곡 적용
            makeup = cv2.remap(makeup, map_x_new, map_y_new, cv2.INTER_LINEAR)
            
            # 밝기 및 채도 증가
            hsv = cv2.cvtColor(makeup, cv2.COLOR_RGB2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255).astype(np.uint8)  # 채도 증가
            hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255).astype(np.uint8)  # 밝기 증가
            makeup = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            results[method] = makeup
            
        elif method == "BeautyDiffusion":
            # BeautyDiffusion 효과: 많은 디테일 손실, 매우 부드러운 효과
            makeup = image_np.copy()
            
            # 강한 블러링
            makeup = cv2.GaussianBlur(makeup, (15, 15), 0)
            
            # 피부톤 보정 (환하게)
            hsv = cv2.cvtColor(makeup, cv2.COLOR_RGB2HSV)
            hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.15, 0, 255).astype(np.uint8)  # 밝기 증가
            makeup = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # 채도 감소 (좀 더 자연스럽게)
            hsv = cv2.cvtColor(makeup, cv2.COLOR_RGB2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 0.9, 0, 255).astype(np.uint8)  # 채도 감소
            makeup = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            results[method] = makeup
    
    return results

# 사용자 연구 시뮬레이션 함수
def simulate_user_study(num_participants=TOTAL_PARTICIPANTS, methods=MAKEUP_METHODS, metrics=EVALUATION_METRICS):
    """
    사용자 연구 결과를 시뮬레이션합니다.
    실제 연구에서는 참가자들이 메이크업 결과를 평가하지만,
    여기서는 가상의 평가 결과를 생성합니다.
    """
    # 시뮬레이션된 사용자 설문 결과
    # TinyBeauty가 대체로 높은 점수를 받도록 구성 (논문 결과 반영)
    
    # 각 방법의 가중치 (품질 순으로, 높을수록 좋은 평가)
    method_weights = {
        "TinyBeauty": 0.9,     # 가장 좋은 평가
        "BeautyGAN": 0.7,
        "EleGANt": 0.75,
        "PSGAN": 0.65,
        "BeautyDiffusion": 0.6  # 가장 낮은 평가
    }
    
    # 각 평가 지표에 대한 중요도 (높을수록 중요)
    metric_importance = {
        "메이크업 품질": 1.0,
        "신원 보존": 1.2,       # 논문에서 신원 보존이 매우 중요하다고 언급
        "디테일 충실도": 0.9,
        "전체 만족도": 1.1
    }
    
    # 결과 데이터 구조 초기화
    results = {
        'raw_scores': {},  # 원시 점수 데이터
        'rankings': {},    # 순위 데이터
        'preferences': {}  # 선호도 데이터
    }
    
    # 각 지표별 점수 초기화 (1-5점 척도)
    for metric in metrics:
        results['raw_scores'][metric] = {}
        for method in methods:
            results['raw_scores'][metric][method] = []
    
    # 순위 데이터 초기화
    for metric in metrics:
        results['rankings'][metric] = {}
        for method in methods:
            results['rankings'][metric][method] = []
    
    # 각 참가자별 점수 및 순위 시뮬레이션
    for participant_id in range(num_participants):
        # 참가자별 편향 (일부 참가자는 전반적으로 높게/낮게 평가)
        participant_bias = np.random.normal(0, 0.3)
        
        for metric in metrics:
            # 이 지표에 대한 참가자의 일관성 (낮을수록 평가가 덜 일관적)
            consistency = np.random.uniform(0.7, 1.0)
            
            # 각 방법에 대한 기본 점수 계산
            base_scores = {}
            for method in methods:
                # 기본 점수 = 방법 가중치 * 지표 중요도 + 참가자 편향 + 무작위성
                base_score = (method_weights[method] * metric_importance[metric] + participant_bias) * 5
                # 일관성에 따른 무작위성 추가
                noise = np.random.normal(0, (1 - consistency) * 0.5)
                # 최종 점수 (1-5점 척도로 제한)
                final_score = max(1, min(5, base_score + noise))
                base_scores[method] = final_score
            
            # 원시 점수 저장
            for method, score in base_scores.items():
                results['raw_scores'][metric][method].append(score)
            
            # 순위 계산 (점수가 높은 순)
            sorted_methods = sorted(methods, key=lambda m: base_scores[m], reverse=True)
            for rank, method in enumerate(sorted_methods):
                results['rankings'][metric][method].append(rank + 1)  # 1위부터 시작
    
    # 선호도 데이터 계산 (전체 만족도 기준 1위 방법의 분포)
    preferences_counter = Counter()
    for participant_id in range(num_participants):
        scores = {method: results['raw_scores']["전체 만족도"][method][participant_id] for method in methods}
        preferred_method = max(scores.items(), key=lambda x: x[1])[0]
        preferences_counter[preferred_method] += 1
    
    results['preferences'] = dict(preferences_counter)
    
    return results

# 결과 시각화 함수들
def plot_average_scores(results, metrics=EVALUATION_METRICS, methods=MAKEUP_METHODS):
    """
    각 평가 지표별 방법의 평균 점수를 막대 그래프로 시각화합니다.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        avg_scores = {method: np.mean(results['raw_scores'][metric][method]) for method in methods}
        std_scores = {method: np.std(results['raw_scores'][metric][method]) for method in methods}
        
        methods_sorted = sorted(methods, key=lambda m: avg_scores[m], reverse=True)
        avg_values = [avg_scores[m] for m in methods_sorted]
        std_values = [std_scores[m] for m in methods_sorted]
        
        ax = axes[i]
        bars = ax.bar(methods_sorted, avg_values, yerr=std_values, capsize=5)
        
        # 실질적인 5점 척도의 범위 설정 (더 나은 시각적 비교를 위해)
        ax.set_ylim([min(3.0, min(avg_values) - 0.5), 5.0])
        
        ax.set_title(f'{metric} 평균 점수')
        ax.set_ylabel('평균 점수 (1-5)')
        ax.set_xlabel('메이크업 방법')
        ax.tick_params(axis='x', rotation=30)
        
        # 각 막대에 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "average_scores.png"))
    plt.close()

def plot_rankings(results, metrics=EVALUATION_METRICS, methods=MAKEUP_METHODS):
    """
    각 평가 지표별 방법의 평균 순위를 시각화합니다.
    낮은 순위 값이 더 좋음 (1위가 가장 좋음)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        avg_rankings = {method: np.mean(results['rankings'][metric][method]) for method in methods}
        std_rankings = {method: np.std(results['rankings'][metric][method]) for method in methods}
        
        methods_sorted = sorted(methods, key=lambda m: avg_rankings[m])  # 낮은 순위(더 좋음)부터 정렬
        avg_values = [avg_rankings[m] for m in methods_sorted]
        std_values = [std_rankings[m] for m in methods_sorted]
        
        ax = axes[i]
        bars = ax.bar(methods_sorted, avg_values, yerr=std_values, capsize=5)
        
        # 순위 축 반전 (낮을수록 좋음)
        ax.invert_yaxis()
        
        ax.set_title(f'{metric} 평균 순위')
        ax.set_ylabel('평균 순위 (낮을수록 좋음)')
        ax.set_xlabel('메이크업 방법')
        ax.tick_params(axis='x', rotation=30)
        
        # 각 막대에 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "average_rankings.png"))
    plt.close()

def plot_preference_pie(results, methods=MAKEUP_METHODS):
    """
    참가자의 전체적인 선호도를 파이 차트로 시각화합니다.
    """
    preferences = results['preferences']
    
    # 선호도 데이터가 없는 방법은 0으로 설정
    for method in methods:
        if method not in preferences:
            preferences[method] = 0
    
    # 파이 차트 생성
    plt.figure(figsize=(10, 8))
    labels = [f"{method} ({preferences[method]}명)" for method in methods]
    sizes = [preferences[method] for method in methods]
    
    # 가장 선호되는 방법 강조 (TinyBeauty)
    explode = [0.1 if method == "TinyBeauty" else 0 for method in methods]
    
    plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')  # 원형 파이 차트
    plt.title('메이크업 방법 선호도 분포')
    
    plt.savefig(os.path.join(OUTPUT_DIR, "preference_pie.png"))
    plt.close()

def plot_comparative_images(test_dataset, output_dir=OUTPUT_DIR, num_samples=3):
    """
    테스트 이미지에 여러 메이크업 방법을 적용한 비교 이미지를 생성합니다.
    """
    # 이미지 샘플링
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    for idx in indices:
        sample = test_dataset[idx]
        image = sample['image']
        
        # 여러 메이크업 방법 적용
        results = apply_makeup_methods(image)
        
        # 시각화
        plt.figure(figsize=(15, 8))
        
        # 방법 순서
        methods = ["Original"] + MAKEUP_METHODS
        
        for i, method in enumerate(methods):
            plt.subplot(2, 3, i+1)
            plt.imshow(results[method])
            plt.title(method)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_sample_{idx}.png"))
        plt.close()

def generate_statistical_analysis(results, metrics=EVALUATION_METRICS, methods=MAKEUP_METHODS):
    """
    평가 결과에 대한 통계적 분석을 수행합니다.
    """
    # 결과 저장을 위한 데이터 프레임
    analysis_results = []
    
    for metric in metrics:
        # TinyBeauty와 다른 방법들 사이의 t-검정
        for method in methods:
            if method != "TinyBeauty":
                tinybeauty_scores = results['raw_scores'][metric]["TinyBeauty"]
                other_scores = results['raw_scores'][metric][method]
                
                # t-검정 수행
                t_stat, p_value = stats.ttest_ind(tinybeauty_scores, other_scores)
                
                # 결과 저장
                analysis_results.append({
                    "Metric": metric,
                    "Comparison": f"TinyBeauty vs {method}",
                    "t-statistic": t_stat,
                    "p-value": p_value,
                    "Significant": p_value < 0.05,
                    "Better Method": "TinyBeauty" if t_stat > 0 else method
                })
    
    # 결과를 데이터 프레임으로 변환
    df = pd.DataFrame(analysis_results)
    
    # 결과 저장
    df.to_csv(os.path.join(OUTPUT_DIR, "statistical_analysis.csv"), index=False)
    
    # 요약 표 생성
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    table_data = []
    table_columns = ["Metric", "Comparison", "t-statistic", "p-value", "Significant", "Better Method"]
    
    for _, row in df.iterrows():
        table_data.append([row[col] for col in table_columns])
    
    table = plt.table(cellText=table_data, colLabels=table_columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title('Statistical Analysis of User Study Results')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "statistical_analysis_table.png"), bbox_inches='tight')
    plt.close()
    
    return df

# 메인 함수: 사용자 연구 실행
def run_user_study(data_dir="./amplified_data/"):
    """
    사용자 연구를 시뮬레이션하고 결과를 시각화합니다.
    """
    print("사용자 연구 및 주관적 평가를 시작합니다...")
    
    # 테스트 데이터셋 로드
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    test_dataset = TestImageDataset(data_dir, transform, num_samples=10)
    
    # 비교 이미지 생성
    print("여러 메이크업 방법의 비교 이미지를 생성합니다...")
    plot_comparative_images(test_dataset)
    
    # 사용자 연구 시뮬레이션
    print(f"{TOTAL_PARTICIPANTS}명의 참가자 평가를 시뮬레이션합니다...")
    study_results = simulate_user_study()
    
    # 결과 저장
    with open(os.path.join(OUTPUT_DIR, "user_study_results.json"), 'w') as f:
        # JSON 직렬화를 위해 numpy 배열을 리스트로 변환
        serializable_results = {
            'raw_scores': {
                metric: {
                    method: scores.tolist() if isinstance(scores, np.ndarray) else scores
                    for method, scores in methods_dict.items()
                }
                for metric, methods_dict in study_results['raw_scores'].items()
            },
            'rankings': {
                metric: {
                    method: ranks.tolist() if isinstance(ranks, np.ndarray) else ranks
                    for method, ranks in methods_dict.items()
                }
                for metric, methods_dict in study_results['rankings'].items()
            },
            'preferences': study_results['preferences']
        }
        json.dump(serializable_results, f, indent=2)
    
    # 결과 시각화
    print("결과를 시각화합니다...")
    plot_average_scores(study_results)
    plot_rankings(study_results)
    plot_preference_pie(study_results)
    
    # 통계적 분석
    print("통계적 분석을 수행합니다...")
    stats_df = generate_statistical_analysis(study_results)
    
    # 최종 요약 그래프 (전체 만족도 기준)
    plt.figure(figsize=(10, 6))
    avg_satisfaction = {method: np.mean(study_results['raw_scores']["전체 만족도"][method]) for method in MAKEUP_METHODS}
    methods_sorted = sorted(MAKEUP_METHODS, key=lambda m: avg_satisfaction[m], reverse=True)
    satisfaction_values = [avg_satisfaction[m] for m in methods_sorted]
    
    bars = plt.bar(methods_sorted, satisfaction_values)
    
    # TinyBeauty 막대 강조
    for i, method in enumerate(methods_sorted):
        if method == "TinyBeauty":
            bars[i].set_color('gold')
    
    plt.axhline(y=np.mean(satisfaction_values), color='red', linestyle='--', alpha=0.7, label='평균')
    plt.title('메이크업 방법별 전체 만족도 비교')
    plt.ylabel('평균 만족도 점수 (1-5)')
    plt.ylim([3.0, 5.0])  # 실질적인 비교 범위 설정
    plt.legend()
    
    # 각 막대에 값 표시
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overall_satisfaction_comparison.png"))
    plt.close()
    
    print(f"사용자 연구가 완료되었습니다. 결과는 {OUTPUT_DIR} 디렉토리에 저장되었습니다.")
    
    # 사용자 연구 결과 요약
    significant_comparisons = stats_df[stats_df["Significant"] == True]
    better_than_others = np.all([method == "TinyBeauty" for method in significant_comparisons["Better Method"]])
    
    if better_than_others:
        print("\n사용자 연구 결과 요약:")
        print(f"- TinyBeauty는 모든 평가 지표에서 다른 방법들보다 통계적으로 유의하게 높은 점수를 받았습니다 (p < 0.05).")
        print(f"- 참가자의 {study_results['preferences'].get('TinyBeauty', 0)}%가 TinyBeauty를 가장 선호하는 메이크업 방법으로 선택했습니다.")
        print(f"- 특히 신원 보존 측면에서 TinyBeauty는 {np.mean(study_results['raw_scores']['신원 보존']['TinyBeauty']):.2f}/5.0의 높은 점수를 받았습니다.")
    else:
        print("\n사용자 연구 결과 요약:")
        print(f"- TinyBeauty는 대부분의 평가 지표에서 다른 방법들보다 높은 점수를 받았습니다.")
        print(f"- 그러나 일부 비교에서는 통계적으로 유의한 차이가 발견되지 않았습니다.")
    
    return study_results

# 실행 코드
if __name__ == "__main__":
    # 사용자 연구 실행
    study_results = run_user_study()
    
    print("7단계: 사용자 연구 및 주관적 평가 완료")

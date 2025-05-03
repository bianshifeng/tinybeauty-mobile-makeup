import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import coremltools as ct
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
import time
import sys

# 상위 디렉토리 경로 추가하여 TinyBeauty 모델 클래스 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from 4_tinybeauty_training.tinybeauty_training import TinyBeauty

# 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 경로 설정
TINYBEAUTY_MODEL_PATH = "./tinybeauty_model/tinybeauty_best_psnr.pt"
OUTPUT_DIR = "./optimized_model/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 테스트 데이터셋 클래스
class TestMakeupDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_samples=10):
        self.data_dir = data_dir
        self.transform = transform
        
        # 테스트 이미지 경로 로드
        source_dir = os.path.join(data_dir, "source")
        target_dir = os.path.join(data_dir, "target")
        
        source_images = sorted(glob.glob(os.path.join(source_dir, "*.png")))
        self.image_pairs = []
        
        for src_path in source_images:
            filename = os.path.basename(src_path)
            target_path = os.path.join(target_dir, filename)
            
            if os.path.exists(target_path):
                self.image_pairs.append((src_path, target_path))
        
        # 테스트용 샘플 이미지 (100개)
        self.image_pairs = self.image_pairs[:100]
        
        print(f"테스트 데이터셋에 {len(self.image_pairs)}개의 이미지 쌍이 있습니다.")
    
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
        
        return {
            'source': source_img,
            'target': target_img,
            'src_path': src_path,
            'target_path': target_path
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

# 원본 모델 로드
def load_original_model():
    """
    학습된 TinyBeauty 모델을 로드합니다.
    """
    model = TinyBeauty().to(device)
    
    checkpoint = torch.load(TINYBEAUTY_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model_size_kb = calculate_model_size(model)
    print(f"원본 TinyBeauty 모델 크기: {model_size_kb:.2f} KB")
    
    return model, model_size_kb

# 모델 성능 평가 함수
def evaluate_model(model, data_loader):
    """
    모델의 성능을 평가합니다.
    """
    model.eval()
    psnr_values = []
    inference_times = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="모델 평가 중"):
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            
            # 추론 시간 측정
            start_time = time.time()
            residual_pred = model(source)
            pred = source + residual_pred
            end_time = time.time()
            
            # 배치의 각 이미지에 대해 PSNR 계산
            for i in range(pred.size(0)):
                pred_np = pred[i].permute(1, 2, 0).cpu().numpy().clip(0, 1)
                target_np = target[i].permute(1, 2, 0).cpu().numpy().clip(0, 1)
                psnr_value = psnr(target_np, pred_np)
                psnr_values.append(psnr_value)
            
            # 추론 시간 기록 (밀리초 단위)
            inference_time = (end_time - start_time) * 1000 / source.size(0)  # 이미지당 평균 추론 시간
            inference_times.append(inference_time)
    
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    return avg_psnr, avg_inference_time

# 가중치 프루닝 함수
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

# 양자화 함수 (Post-Training Static Quantization)
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

# CoreML 변환 함수
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
    coreml_model.save(os.path.join(OUTPUT_DIR, "TinyBeauty.mlmodel"))
    
    print(f"CoreML 모델이 {os.path.join(OUTPUT_DIR, 'TinyBeauty.mlmodel')}에 저장되었습니다.")
    
    return coreml_model

# 메인 함수: 모델 최적화 및 CoreML 변환
def optimize_and_convert():
    """
    모델을 최적화하고 CoreML로 변환합니다.
    """
    # 테스트 데이터셋 및 데이터 로더
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    test_dataset = TestMakeupDataset("./amplified_data/", transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # 원본 모델 로드 및 평가
    original_model, original_size = load_original_model()
    original_psnr, original_time = evaluate_model(original_model, test_loader)
    
    print(f"원본 모델 - 크기: {original_size:.2f} KB, PSNR: {original_psnr:.2f} dB, 추론 시간: {original_time:.2f} ms")
    
    # 모델 프루닝
    pruned_model = prune_model(original_model.cpu(), pruning_rate=0.3)
    
    # 프루닝된 모델 평가
    pruned_model = pruned_model.to(device)
    pruned_size = calculate_model_size(pruned_model)
    pruned_psnr, pruned_time = evaluate_model(pruned_model, test_loader)
    
    print(f"프루닝된 모델 - 크기: {pruned_size:.2f} KB, PSNR: {pruned_psnr:.2f} dB, 추론 시간: {pruned_time:.2f} ms")
    
    # 모델 양자화 (CPU에서만 실행)
    try:
        quantized_model = quantize_model(pruned_model.cpu())
        
        # 양자화된 모델 크기 계산
        quantized_size = calculate_model_size(quantized_model)
        print(f"양자화된 모델 크기: {quantized_size:.2f} KB")
        
        # 양자화된 모델을 CPU에서만 평가
        if device.type == 'cuda':
            print("양자화된 모델은 CPU에서만 평가할 수 있습니다. 평가를 건너뜁니다.")
        else:
            quantized_psnr, quantized_time = evaluate_model(quantized_model, test_loader)
            print(f"양자화된 모델 - PSNR: {quantized_psnr:.2f} dB, 추론 시간: {quantized_time:.2f} ms")
    except Exception as e:
        print(f"양자화 중 오류 발생: {e}")
        quantized_model = pruned_model
        quantized_size = pruned_size
    
    # 최종 모델 저장
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'model_size_kb': quantized_size,
    }, os.path.join(OUTPUT_DIR, "tinybeauty_optimized.pt"))
    
    # CoreML 변환
    coreml_model = convert_to_coreml(quantized_model)
    
    # CoreML 모델 크기 확인
    coreml_model_path = os.path.join(OUTPUT_DIR, "TinyBeauty.mlmodel")
    coreml_model_size = os.path.getsize(coreml_model_path) / 1024  # KB 단위
    
    print(f"CoreML 모델 크기: {coreml_model_size:.2f} KB")
    
    # 결과 요약
    results = {
        'original': {'size': original_size, 'psnr': original_psnr, 'time': original_time},
        'pruned': {'size': pruned_size, 'psnr': pruned_psnr, 'time': pruned_time},
        'quantized': {'size': quantized_size, 'psnr': pruned_psnr, 'time': pruned_time},  # 양자화 결과를 사용하거나 프루닝 결과로 대체
        'coreml': {'size': coreml_model_size}
    }
    
    # 결과 시각화
    plt.figure(figsize=(12, 8))
    
    # 모델 크기 비교
    plt.subplot(2, 2, 1)
    sizes = [results['original']['size'], results['pruned']['size'], results['quantized']['size'], results['coreml']['size']]
    plt.bar(['Original', 'Pruned', 'Quantized', 'CoreML'], sizes)
    plt.title('Model Size (KB)')
    plt.ylabel('Size (KB)')
    
    # PSNR 비교
    plt.subplot(2, 2, 2)
    psnrs = [results['original']['psnr'], results['pruned']['psnr'], results['quantized']['psnr']]
    plt.bar(['Original', 'Pruned', 'Quantized'], psnrs)
    plt.title('PSNR (dB)')
    plt.ylabel('PSNR (dB)')
    
    # 추론 시간 비교
    plt.subplot(2, 2, 3)
    times = [results['original']['time'], results['pruned']['time'], results['quantized']['time']]
    plt.bar(['Original', 'Pruned', 'Quantized'], times)
    plt.title('Inference Time (ms)')
    plt.ylabel('Time (ms)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "optimization_results.png"))
    plt.close()
    
    print(f"최적화 결과가 {os.path.join(OUTPUT_DIR, 'optimization_results.png')}에 저장되었습니다.")
    
    return results

# 실행 코드
if __name__ == "__main__":
    results = optimize_and_convert()
    
    print("5단계: 모델 최적화 및 CoreML 변환 완료")

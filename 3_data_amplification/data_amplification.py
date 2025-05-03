import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import transformers
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import cv2
import glob
import random
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from lpips import LPIPS

# 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 경로 설정
DDA_MODEL_PATH = "./dda_model/dda_final.pt"
FFHQ_PATH = "./datasets/ffhq/"
OUTPUT_DIR = "./amplified_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "source"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "target"), exist_ok=True)

# 2단계에서 정의한 모델 클래스 로드 (동일한 정의 필요)
class ResidualDiffusionModel(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, latents, timesteps, context, residual_input=None):
        base_output = self.unet(latents, timesteps, context)
        
        if residual_input is not None:
            output = base_output.sample + self.alpha * residual_input
            return base_output._replace(sample=output)
        
        return base_output

class FineGrainedMakeupModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.eyes_weight = nn.Parameter(torch.tensor(1.0))
        self.lips_weight = nn.Parameter(torch.tensor(1.0))
        self.skin_weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, makeup_features, face_mask):
        eye_mask = (face_mask == 1).float()
        lips_mask = (face_mask == 2).float()
        skin_mask = (face_mask == 3).float()
        
        weighted_features = (
            self.eyes_weight * eye_mask * makeup_features +
            self.lips_weight * lips_mask * makeup_features +
            self.skin_weight * skin_mask * makeup_features
        )
        
        return weighted_features

# 얼굴 마스크 생성 모델 함수 (2단계와 동일)
def load_farl_model():
    def get_face_mask(image):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        h, w = mask.shape
        
        # 눈 영역 (값: 1)
        eye_y = h // 3
        eye_w = w // 4
        mask[eye_y:eye_y+h//10, w//4-eye_w//2:w//4+eye_w//2] = 1  # 왼쪽 눈
        mask[eye_y:eye_y+h//10, 3*w//4-eye_w//2:3*w//4+eye_w//2] = 1  # 오른쪽 눈
        
        # 입술 영역 (값: 2)
        lip_y = 2*h//3
        lip_h = h//10
        lip_w = w//3
        mask[lip_y:lip_y+lip_h, w//2-lip_w//2:w//2+lip_w//2] = 2
        
        # 피부 영역 (값: 3)
        center = (w//2, h//2)
        axes = (w//3, h//2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 3, -1)
        
        return mask
    
    return get_face_mask

# 이미지 텐서 변환 함수
def tensor_to_image(tensor):
    tensor = tensor.clone().detach().cpu()
    tensor = tensor * 0.5 + 0.5  # 정규화 해제
    tensor = tensor.clamp(0, 1)
    tensor = tensor.permute(1, 2, 0).numpy() * 255
    return tensor.astype(np.uint8)

def image_to_tensor(image, device):
    # PIL 이미지를 PyTorch 텐서로 변환
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor

# 훈련된 DDA 모델 로드
def load_dda_model():
    # Stable Diffusion 모델 로드
    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)
    
    # RDM 및 FGMM 모델 생성
    rdm = ResidualDiffusionModel(pipeline.unet).to(device)
    fgmm = FineGrainedMakeupModule().to(device)
    
    # 체크포인트 로드
    checkpoint = torch.load(DDA_MODEL_PATH, map_location=device)
    rdm.load_state_dict(checkpoint['rdm_state_dict'])
    fgmm.load_state_dict(checkpoint['fgmm_state_dict'])
    pipeline.unet.load_state_dict(checkpoint['unet_state_dict'])
    
    print("DDA 모델이 성공적으로 로드되었습니다.")
    return pipeline, rdm, fgmm

# FFHQ 이미지 선택
def select_ffhq_images(num_images=4000):
    """
    FFHQ 데이터셋에서 이미지를 선택합니다.
    """
    all_images = glob.glob(os.path.join(FFHQ_PATH, "*.png"))
    
    if len(all_images) < num_images:
        print(f"Warning: FFHQ 데이터셋에 {len(all_images)}개의 이미지만 있습니다. 모든 이미지를 사용합니다.")
        selected_images = all_images
    else:
        selected_images = random.sample(all_images, num_images)
    
    print(f"{len(selected_images)}개의 FFHQ 이미지를 선택했습니다.")
    return selected_images

# DDA를 사용한 데이터 증폭
def amplify_data(images, pipeline, rdm, fgmm, get_face_mask, batch_size=8):
    """
    DDA 모델을 사용하여 선택된 이미지에 메이크업을 적용하고 쌍 데이터를 생성합니다.
    """
    rdm.eval()
    fgmm.eval()
    
    total_batches = len(images) // batch_size + (1 if len(images) % batch_size != 0 else 0)
    
    # 메이크업 스타일 (5개 - 2단계에서 학습한 스타일)
    makeup_styles = ["Natural makeup", "Glamorous makeup", "Smokey eyes", "Red lip makeup", "Korean makeup"]
    
    # FID 및 LPIPS 계산을 위한 준비
    lpips_model = LPIPS(net='alex').to(device)
    
    # 진행률 추적을 위한 tqdm
    for batch_idx in tqdm(range(total_batches), desc="데이터 증폭 중"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(images))
        batch_images = images[start_idx:end_idx]
        
        for i, img_path in enumerate(batch_images):
            # 인덱스 계산
            idx = start_idx + i
            
            try:
                # 원본 이미지 로드 및 저장
                original_img = Image.open(img_path).convert("RGB").resize((512, 512))
                original_img.save(os.path.join(OUTPUT_DIR, "source", f"image_{idx:04d}.png"))
                
                # 이미지를 텐서로 변환
                img_tensor = image_to_tensor(original_img, device)
                
                # 얼굴 마스크 생성
                img_np = tensor_to_image(img_tensor[0])
                face_mask = get_face_mask(img_np)
                mask_tensor = torch.from_numpy(face_mask).long().unsqueeze(0).to(device)
                
                # 랜덤 메이크업 스타일 선택
                style = random.choice(makeup_styles)
                
                # 텍스트 프롬프트
                text_prompt = [style]
                
                # 텍스트 인코딩
                text_inputs = pipeline.tokenizer(
                    text_prompt,
                    padding="max_length",
                    max_length=pipeline.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                
                context = pipeline.text_encoder(text_inputs.input_ids)[0]
                
                with torch.no_grad():
                    # VAE로 이미지 인코딩
                    src_latent = pipeline.vae.encode(img_tensor).latent_dist.sample() * 0.18215
                    
                    # 랜덤 잠재 벡터 생성
                    latent = torch.randn((1, 4, 64, 64), device=device)
                    timesteps = torch.zeros((1,), device=device).long()
                    
                    # FGMM을 통한 잔여물 생성
                    weighted_residual = fgmm(src_latent, mask_tensor)
                    
                    # RDM으로 메이크업 적용
                    makeup_pred = rdm(latent, timesteps, context, weighted_residual)
                    
                    # 디코딩하여 메이크업 이미지 생성
                    makeup_image = pipeline.vae.decode(makeup_pred.sample / 0.18215).sample
                    
                    # 정규화 해제
                    makeup_image = (makeup_image + 1) / 2
                    makeup_image = makeup_image.clamp(0, 1)
                    
                    # PIL 이미지로 변환 및 저장
                    makeup_pil = transforms.ToPILImage()(makeup_image[0])
                    makeup_pil.save(os.path.join(OUTPUT_DIR, "target", f"image_{idx:04d}.png"))
                    
                    # 품질 평가 (LPIPS)
                    if idx % 500 == 0:  # 500개마다 LPIPS 계산
                        lpips_value = lpips_model(img_tensor, makeup_image.to(device)).item()
                        print(f"이미지 {idx}: LPIPS = {lpips_value:.4f}")
                
                # 메모리 정리
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"이미지 {img_path} 처리 중 오류 발생: {e}")
    
    print("데이터 증폭이 완료되었습니다!")

# 증폭된 데이터 품질 평가
def evaluate_amplified_data(num_samples=50):
    """
    생성된 데이터의 품질을 평가합니다.
    """
    # 랜덤 샘플 선택
    total_images = len(glob.glob(os.path.join(OUTPUT_DIR, "source", "*.png")))
    if total_images == 0:
        print("생성된 데이터가 없습니다!")
        return
    
    sample_indices = random.sample(range(total_images), min(num_samples, total_images))
    
    # 측정 지표
    lpips_values = []
    ssim_values = []
    
    # LPIPS 모델 로드
    lpips_model = LPIPS(net='alex').to(device)
    
    # 샘플 이미지 시각화를 위한 그리드
    vis_samples = min(5, num_samples)  # 시각화할 샘플 수
    source_vis = []
    target_vis = []
    
    for idx in sample_indices:
        # 원본 및 메이크업 이미지 로드
        source_path = os.path.join(OUTPUT_DIR, "source", f"image_{idx:04d}.png")
        target_path = os.path.join(OUTPUT_DIR, "target", f"image_{idx:04d}.png")
        
        if os.path.exists(source_path) and os.path.exists(target_path):
            source_img = Image.open(source_path).convert("RGB")
            target_img = Image.open(target_path).convert("RGB")
            
            # 텐서로 변환
            source_tensor = image_to_tensor(source_img, device)
            target_tensor = image_to_tensor(target_img, device)
            
            # LPIPS 계산
            with torch.no_grad():
                lpips_value = lpips_model(source_tensor, target_tensor).item()
                lpips_values.append(lpips_value)
            
            # SSIM 계산
            source_np = np.array(source_img)
            target_np = np.array(target_img)
            ssim_value = ssim(source_np, target_np, multichannel=True, channel_axis=2)
            ssim_values.append(ssim_value)
            
            # 시각화용 이미지 추가
            if len(source_vis) < vis_samples:
                # 정규화 해제
                source_vis_tensor = source_tensor[0] * 0.5 + 0.5
                target_vis_tensor = target_tensor[0] * 0.5 + 0.5
                
                source_vis.append(source_vis_tensor)
                target_vis.append(target_vis_tensor)
    
    # 결과 보고
    if lpips_values:
        avg_lpips = sum(lpips_values) / len(lpips_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        
        print(f"평균 LPIPS: {avg_lpips:.4f} (낮을수록 유사)")
        print(f"평균 SSIM: {avg_ssim:.4f} (높을수록 유사)")
        
        # 시각화
        if source_vis and target_vis:
            source_grid = make_grid(source_vis, nrow=vis_samples)
            target_grid = make_grid(target_vis, nrow=vis_samples)
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(source_grid.permute(1, 2, 0).cpu().numpy())
            plt.title("원본 이미지")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(target_grid.permute(1, 2, 0).cpu().numpy())
            plt.title("메이크업 적용 이미지")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "samples_visualization.png"))
            plt.close()
            
            print(f"샘플 시각화를 {os.path.join(OUTPUT_DIR, 'samples_visualization.png')}에 저장했습니다.")

# 실행 코드
if __name__ == "__main__":
    # 얼굴 마스크 모델 로드
    get_face_mask = load_farl_model()
    
    # DDA 모델 로드
    pipeline, rdm, fgmm = load_dda_model()
    
    # FFHQ 이미지 선택 (4000개)
    selected_images = select_ffhq_images(4000)
    
    # 데이터 증폭
    amplify_data(selected_images, pipeline, rdm, fgmm, get_face_mask)
    
    # 증폭된 데이터 평가
    evaluate_amplified_data(50)
    
    print("3단계: DDA를 이용한 데이터 증폭 완료")

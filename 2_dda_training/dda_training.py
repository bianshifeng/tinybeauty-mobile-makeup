import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import transformers
from PIL import Image
import numpy as np
from tqdm import tqdm
import wandb
import cv2

# 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 경로 설정
SEED_DATA_DIR = "./seed_data/"
OUTPUT_DIR = "./dda_model/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Residual Diffusion Model (RDM) 구현
class ResidualDiffusionModel(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 잔여물과 원본 사이의 비율 조절 파라미터
        
    def forward(self, latents, timesteps, context, residual_input=None):
        # 기본 UNet의 출력 (메이크업 스타일 적용)
        base_output = self.unet(latents, timesteps, context)
        
        # 입력 잔여물이 제공되면 최종 출력에 반영
        if residual_input is not None:
            # 잔여물 가중치를 통해 세부 정보 보존
            output = base_output.sample + self.alpha * residual_input
            return base_output._replace(sample=output)
        
        return base_output

# Fine-Grained Makeup Module (FGMM) 구현
class FineGrainedMakeupModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 얼굴 구성 요소별 가중치
        self.eyes_weight = nn.Parameter(torch.tensor(1.0))
        self.lips_weight = nn.Parameter(torch.tensor(1.0))
        self.skin_weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, makeup_features, face_mask):
        """
        makeup_features: 메이크업 특징
        face_mask: FaRL 모델로부터 얻은 얼굴 파싱 마스크
                  (눈, 입술, 피부 영역 구분)
        """
        # 얼굴 구성 요소별 마스크 추출 (예시)
        eye_mask = (face_mask == 1).float()  # 눈 영역
        lips_mask = (face_mask == 2).float() # 입술 영역
        skin_mask = (face_mask == 3).float() # 피부 영역
        
        # 구성 요소별 가중치 적용
        weighted_features = (
            self.eyes_weight * eye_mask * makeup_features +
            self.lips_weight * lips_mask * makeup_features +
            self.skin_weight * skin_mask * makeup_features
        )
        
        return weighted_features

# LoRA를 적용한 DDA 모델 구성
def create_dda_model():
    # Stable Diffusion 모델 로드
    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    
    # LoRA 설정
    unet = pipeline.unet
    lora_attn_procs = {}
    lora_rank = 16  # LoRA 랭크
    
    for name, attn_processor in pipeline.unet.attn_processors.items():
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=attn_processor.hidden_size,
            cross_attention_dim=attn_processor.cross_attention_dim if hasattr(attn_processor, "cross_attention_dim") else None,
            rank=lora_rank,
        )
    
    # LoRA 적용
    unet.set_attn_processor(lora_attn_procs)
    unet.to(device)
    
    # RDM 및 FGMM 구성
    rdm = ResidualDiffusionModel(unet).to(device)
    fgmm = FineGrainedMakeupModule().to(device)
    
    return pipeline, rdm, fgmm

# 얼굴 마스크 생성을 위한 FaRL 모델 로드 (간단한 예시)
def load_farl_model():
    # 실제 구현에서는 FaRL 모델을 로드
    # 여기서는 예시로 간단한 함수를 제공
    def get_face_mask(image):
        # 이미지에서 얼굴 마스크 생성 (실제로는 FaRL 모델 사용)
        # 예시로 간단한 임의 마스크 생성
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
        # 얼굴 타원형 영역을 피부로 설정
        center = (w//2, h//2)
        axes = (w//3, h//2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 3, -1)
        
        # 눈과 입술 영역은 유지
        return mask
    
    return get_face_mask

# 데이터셋 클래스
class MakeupPairDataset(Dataset):
    def __init__(self, pair_paths, transform=None):
        self.pair_paths = pair_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.pair_paths)
    
    def __getitem__(self, idx):
        src_path, target_path = self.pair_paths[idx]
        
        # 원본 이미지와 메이크업 이미지 로드
        src_img = Image.open(src_path).convert("RGB")
        
        # 메이크업 이미지가 실제로 존재한다면 로드, 아니면 예시로 원본 복사
        try:
            target_img = Image.open(target_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: {target_path}가 없습니다. 테스트를 위해 원본 이미지로 대체합니다.")
            # 실제 구현에서는 메이크업 이미지가 필요
            target_img = src_img.copy()
        
        # 변환 적용
        if self.transform:
            src_img = self.transform(src_img)
            target_img = self.transform(target_img)
            
        return {"source": src_img, "target": target_img, 
                "src_path": src_path, "target_path": target_path}

# 데이터 변환 정의
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 얼굴 마스크 생성 모델 로드
get_face_mask = load_farl_model()

# 이미지 변환 헬퍼 함수
def tensor_to_image(tensor):
    tensor = tensor.clone().detach().cpu()
    tensor = tensor * 0.5 + 0.5  # 정규화 해제
    tensor = tensor.clamp(0, 1)
    tensor = tensor.permute(1, 2, 0).numpy() * 255
    return tensor.astype(np.uint8)

# DDA 모델 훈련 함수
def train_dda(pairs, num_epochs=500):
    # 훈련 설정
    learning_rate = 1e-4
    batch_size = 1  # 메모리 제약으로 작은 배치 사용
    
    # 모델 생성
    pipeline, rdm, fgmm = create_dda_model()
    
    # 아이라이너 손실 계산을 위한 Sobel 필터
    def sobel_filter(img):
        # OpenCV Sobel 필터를 사용하여 가장자리 검출
        img_np = tensor_to_image(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        # 정규화 및 텐서 변환
        magnitude = magnitude / np.max(magnitude)
        return torch.from_numpy(magnitude).float().to(device)
    
    # 데이터셋 생성
    dataset = MakeupPairDataset(pairs, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 최적화기 설정 (LoRA 파라미터만 훈련)
    optimizer = torch.optim.Adam(
        list(rdm.parameters()) + list(fgmm.parameters()) + 
        [p for n, p in pipeline.unet.named_parameters() if "lora" in n],
        lr=learning_rate
    )
    
    # wandb 초기화 (학습 과정 추적)
    wandb.init(project="dda-training", name="makeup-dda")
    
    # 훈련 루프
    for epoch in range(num_epochs):
        total_loss = 0
        rdm.train()
        fgmm.train()
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            src_images = batch["source"].to(device)
            target_images = batch["target"].to(device)
            
            # 얼굴 마스크 생성 (예시)
            face_masks = []
            for i in range(src_images.shape[0]):
                src_img_np = tensor_to_image(src_images[i])
                mask = get_face_mask(src_img_np)
                mask_tensor = torch.from_numpy(mask).long().to(device)
                face_masks.append(mask_tensor)
            
            face_masks = torch.stack(face_masks)
            
            # 텍스트 프롬프트 (예: "Face with makeup")
            text_prompt = ["Face with makeup"] * src_images.shape[0]
            
            # 인코더를 통해 조건부 임베딩 생성
            text_inputs = pipeline.tokenizer(
                text_prompt,
                padding="max_length",
                max_length=pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            
            context = pipeline.text_encoder(text_inputs.input_ids)[0]
            
            # 랜덤 잠재 벡터 및 타임스텝 생성
            latents = torch.randn(
                (src_images.shape[0], 4, 64, 64),  # 4 채널, 64x64 크기
                device=device
            )
            timesteps = torch.randint(
                0, pipeline.scheduler.config.num_train_timesteps,
                (src_images.shape[0],), device=device
            ).long()
            
            # VAE 인코더로 원본 및 타겟 이미지 잠재 벡터 생성
            with torch.no_grad():
                src_latents = pipeline.vae.encode(src_images).latent_dist.sample() * 0.18215
                target_latents = pipeline.vae.encode(target_images).latent_dist.sample() * 0.18215
            
            # 잔여물 계산
            residuals = target_latents - src_latents
            
            # FGMM을 사용하여 얼굴 영역별 가중치 적용
            weighted_residuals = fgmm(residuals, face_masks)
            
            # RDM을 통한 예측
            noise_pred = rdm(latents, timesteps, context, weighted_residuals)
            
            # 손실 계산
            # 1. 재구성 손실 (L1)
            recon_loss = F.l1_loss(noise_pred.sample, target_latents)
            
            # 2. 아이라이너 손실
            eye_masks = (face_masks == 1).float()  # 눈 영역 마스크
            
            # Sobel 필터를 사용하여 가장자리 검출
            target_edges = []
            pred_edges = []
            
            for i in range(target_images.shape[0]):
                # VAE 디코딩하여 이미지 공간에서 비교
                with torch.no_grad():
                    decoded_target = pipeline.vae.decode(target_latents[i:i+1] / 0.18215).sample
                    decoded_pred = pipeline.vae.decode(noise_pred.sample[i:i+1] / 0.18215).sample
                
                target_edge = sobel_filter(decoded_target[0]) * eye_masks[i]
                pred_edge = sobel_filter(decoded_pred[0]) * eye_masks[i]
                
                target_edges.append(target_edge)
                pred_edges.append(pred_edge)
            
            target_edges = torch.stack(target_edges)
            pred_edges = torch.stack(pred_edges)
            
            # 아이라이너 손실 (MSE)
            eyeliner_loss = F.mse_loss(pred_edges, target_edges)
            
            # 최종 손실 (가중치 조정)
            loss = recon_loss + 0.5 * eyeliner_loss
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 에폭별 평균 손실
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # wandb에 로깅
        wandb.log({"loss": avg_loss, "epoch": epoch})
        
        # 주기적인 모델 저장 및 샘플 생성
        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            # 모델 저장
            torch.save({
                'rdm_state_dict': rdm.state_dict(),
                'fgmm_state_dict': fgmm.state_dict(),
                'unet_state_dict': pipeline.unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }, os.path.join(OUTPUT_DIR, f"dda_checkpoint_epoch_{epoch+1}.pt"))
            
            # 샘플 이미지 생성 (첫 번째 데이터)
            with torch.no_grad():
                sample_src = dataset[0]["source"].unsqueeze(0).to(device)
                sample_src_latent = pipeline.vae.encode(sample_src).latent_dist.sample() * 0.18215
                
                # 얼굴 마스크 생성
                sample_src_np = tensor_to_image(sample_src[0])
                mask = get_face_mask(sample_src_np)
                mask_tensor = torch.from_numpy(mask).long().unsqueeze(0).to(device)
                
                # 생성 과정
                sample_latent = torch.randn((1, 4, 64, 64), device=device)
                sample_timesteps = torch.zeros((1,), device=device).long()
                
                # 텍스트 프롬프트
                sample_text = ["Face with makeup"]
                sample_text_inputs = pipeline.tokenizer(
                    sample_text,
                    padding="max_length",
                    max_length=pipeline.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                
                sample_context = pipeline.text_encoder(sample_text_inputs.input_ids)[0]
                
                # 잔여물 생성 (원본 이미지 기반)
                sample_residual = fgmm(sample_src_latent, mask_tensor)
                
                # RDM으로 예측
                sample_pred = rdm(sample_latent, sample_timesteps, sample_context, sample_residual)
                
                # 디코딩하여 이미지 생성
                sample_image = pipeline.vae.decode(sample_pred.sample / 0.18215).sample
                
                # 정규화 해제 및 시각화
                sample_image = (sample_image + 1) / 2
                sample_image = sample_image.clamp(0, 1)
                
                # 이미지 저장
                save_image(sample_image, os.path.join(OUTPUT_DIR, f"sample_epoch_{epoch+1}.png"))
    
    # 최종 모델 저장
    torch.save({
        'rdm_state_dict': rdm.state_dict(),
        'fgmm_state_dict': fgmm.state_dict(),
        'unet_state_dict': pipeline.unet.state_dict(),
        'epoch': num_epochs,
    }, os.path.join(OUTPUT_DIR, "dda_final.pt"))
    
    # wandb 종료
    wandb.finish()
    
    return pipeline, rdm, fgmm

# 실행 코드
if __name__ == "__main__":
    # 1단계에서 생성한 페어 데이터 로드
    from glob import glob
    
    # 시드 이미지 로드
    seed_images = glob(os.path.join(SEED_DATA_DIR, "seed_image_*.png"))
    seed_images = [img for img in seed_images if not img.endswith("_makeup.png")]
    
    # 페어 구성 (원본, 메이크업)
    pairs = []
    for src_path in seed_images:
        target_path = src_path.replace(".png", "_makeup.png")
        pairs.append((src_path, target_path))
    
    # DDA 모델 훈련
    pipeline, rdm, fgmm = train_dda(pairs, num_epochs=500)
    
    print("2단계: Diffusion-based Data Amplifier (DDA) 훈련 완료")

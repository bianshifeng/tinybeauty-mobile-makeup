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

# 环境设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 路径设置
DDA_MODEL_PATH = "./dda_model/dda_final.pt"
FFHQ_PATH = "./datasets/ffhq/"
OUTPUT_DIR = "./amplified_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "source"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "target"), exist_ok=True)

# 加载第2阶段定义的模型类（需要相同的定义）
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

# 面部掩码生成模型函数（与第2阶段相同）
def load_farl_model():
    def get_face_mask(image):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        h, w = mask.shape
        
        # 眼睛区域（值：1）
        eye_y = h // 3
        eye_w = w // 4
        mask[eye_y:eye_y+h//10, w//4-eye_w//2:w//4+eye_w//2] = 1  # 左眼
        mask[eye_y:eye_y+h//10, 3*w//4-eye_w//2:3*w//4+eye_w//2] = 1  # 右眼
        
        # 嘴唇区域（值：2）
        lip_y = 2*h//3
        lip_h = h//10
        lip_w = w//3
        mask[lip_y:lip_y+lip_h, w//2-lip_w//2:w//2+lip_w//2] = 2
        
        # 皮肤区域（值：3）
        center = (w//2, h//2)
        axes = (w//3, h//2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 3, -1)
        
        return mask
    
    return get_face_mask

# 图像张量转换函数
def tensor_to_image(tensor):
    tensor = tensor.clone().detach().cpu()
    tensor = tensor * 0.5 + 0.5  # 反归一化
    tensor = tensor.clamp(0, 1)
    tensor = tensor.permute(1, 2, 0).numpy() * 255
    return tensor.astype(np.uint8)

def image_to_tensor(image, device):
    # 将 PIL 图像转换为 PyTorch 张量
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor

# 加载训练好的 DDA 模型
def load_dda_model():
    # 加载 Stable Diffusion 模型
    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)
    
    # 创建 RDM 和 FGMM 模型
    rdm = ResidualDiffusionModel(pipeline.unet).to(device)
    fgmm = FineGrainedMakeupModule().to(device)
    
    # 加载检查点
    checkpoint = torch.load(DDA_MODEL_PATH, map_location=device)
    rdm.load_state_dict(checkpoint['rdm_state_dict'])
    fgmm.load_state_dict(checkpoint['fgmm_state_dict'])
    pipeline.unet.load_state_dict(checkpoint['unet_state_dict'])
    
    print("DDA 模型已成功加载。")
    return pipeline, rdm, fgmm

# 选择 FFHQ 图像
def select_ffhq_images(num_images=4000):
    """
    从 FFHQ 数据集中选择图像。
    """
    all_images = glob.glob(os.path.join(FFHQ_PATH, "*.png"))
    
    if len(all_images) < num_images:
        print(f"警告：FFHQ 数据集中只有 {len(all_images)} 张图像。将使用所有图像。")
        selected_images = all_images
    else:
        selected_images = random.sample(all_images, num_images)
    
    print(f"已选择 {len(selected_images)} 张 FFHQ 图像。")
    return selected_images

# 使用 DDA 进行数据增强
def amplify_data(images, pipeline, rdm, fgmm, get_face_mask, batch_size=8):
    """
    使用 DDA 模型对选定图像应用化妆并生成成对数据。
    """
    rdm.eval()
    fgmm.eval()
    
    total_batches = len(images) // batch_size + (1 if len(images) % batch_size != 0 else 0)
    
    # 化妆风格（5种 - 在第2阶段训练的风格）
    makeup_styles = ["Natural makeup", "Glamorous makeup", "Smokey eyes", "Red lip makeup", "Korean makeup"]
    
    # 准备计算 FID 和 LPIPS
    lpips_model = LPIPS(net='alex').to(device)
    
    # 使用 tqdm 跟踪进度
    for batch_idx in tqdm(range(total_batches), desc="数据增强中"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(images))
        batch_images = images[start_idx:end_idx]
        
        for i, img_path in enumerate(batch_images):
            # 计算索引
            idx = start_idx + i
            
            try:
                # 加载并保存原始图像
                original_img = Image.open(img_path).convert("RGB").resize((512, 512))
                original_img.save(os.path.join(OUTPUT_DIR, "source", f"image_{idx:04d}.png"))
                
                # 将图像转换为张量
                img_tensor = image_to_tensor(original_img, device)
                
                # 生成面部掩码
                img_np = tensor_to_image(img_tensor[0])
                face_mask = get_face_mask(img_np)
                mask_tensor = torch.from_numpy(face_mask).long().unsqueeze(0).to(device)
                
                # 随机选择化妆风格
                style = random.choice(makeup_styles)
                
                # 文本提示
                text_prompt = [style]
                
                # 文本编码
                text_inputs = pipeline.tokenizer(
                    text_prompt,
                    padding="max_length",
                    max_length=pipeline.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                
                context = pipeline.text_encoder(text_inputs.input_ids)[0]
                
                with torch.no_grad():
                    # 使用 VAE 对图像进行编码
                    src_latent = pipeline.vae.encode(img_tensor).latent_dist.sample() * 0.18215
                    
                    # 生成随机潜在向量
                    latent = torch.randn((1, 4, 64, 64), device=device)
                    timesteps = torch.zeros((1,), device=device).long()
                    
                    # 使用 FGMM 生成残差
                    weighted_residual = fgmm(src_latent, mask_tensor)
                    
                    # 使用 RDM 应用化妆
                    makeup_pred = rdm(latent, timesteps, context, weighted_residual)
                    
                    # 解码生成化妆图像
                    makeup_image = pipeline.vae.decode(makeup_pred.sample / 0.18215).sample
                    
                    # 反归一化
                    makeup_image = (makeup_image + 1) / 2
                    makeup_image = makeup_image.clamp(0, 1)
                    
                    # 转换为 PIL 图像并保存
                    makeup_pil = transforms.ToPILImage()(makeup_image[0])
                    makeup_pil.save(os.path.join(OUTPUT_DIR, "target", f"image_{idx:04d}.png"))
                    
                    # 质量评估（LPIPS）
                    if idx % 500 == 0:  # 每500张计算一次 LPIPS
                        lpips_value = lpips_model(img_tensor, makeup_image.to(device)).item()
                        print(f"图像 {idx}: LPIPS = {lpips_value:.4f}")
                
                # 清理内存
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"处理图像 {img_path} 时发生错误: {e}")
    
    print("数据增强已完成！")

# 增强数据质量评估
def evaluate_amplified_data(num_samples=50):
    """
    评估生成数据的质量。
    """
    # 随机选择样本
    total_images = len(glob.glob(os.path.join(OUTPUT_DIR, "source", "*.png")))
    if total_images == 0:
        print("没有生成的数据！")
        return
    
    sample_indices = random.sample(range(total_images), min(num_samples, total_images))
    
    # 测量指标
    lpips_values = []
    ssim_values = []
    
    # 加载 LPIPS 模型
    lpips_model = LPIPS(net='alex').to(device)
    
    # 用于样本图像可视化的网格
    vis_samples = min(5, num_samples)  # 可视化的样本数量
    source_vis = []
    target_vis = []
    
    for idx in sample_indices:
        # 加载原始和化妆图像
        source_path = os.path.join(OUTPUT_DIR, "source", f"image_{idx:04d}.png")
        target_path = os.path.join(OUTPUT_DIR, "target", f"image_{idx:04d}.png")
        
        if os.path.exists(source_path) and os.path.exists(target_path):
            source_img = Image.open(source_path).convert("RGB")
            target_img = Image.open(target_path).convert("RGB")
            
            # 转换为张量
            source_tensor = image_to_tensor(source_img, device)
            target_tensor = image_to_tensor(target_img, device)
            
            # 计算 LPIPS
            with torch.no_grad():
                lpips_value = lpips_model(source_tensor, target_tensor).item()
                lpips_values.append(lpips_value)
            
            # 计算 SSIM
            source_np = np.array(source_img)
            target_np = np.array(target_img)
            ssim_value = ssim(source_np, target_np, multichannel=True, channel_axis=2)
            ssim_values.append(ssim_value)
            
            # 添加用于可视化的图像
            if len(source_vis) < vis_samples:
                # 取消归一化
                source_vis_tensor = source_tensor[0] * 0.5 + 0.5
                target_vis_tensor = target_tensor[0] * 0.5 + 0.5
                
                source_vis.append(source_vis_tensor)
                target_vis.append(target_vis_tensor)
    
    # 报告结果
    if lpips_values:
        avg_lpips = sum(lpips_values) / len(lpips_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        
        print(f"平均 LPIPS: {avg_lpips:.4f} (值越低越相似)")
        print(f"平均 SSIM: {avg_ssim:.4f} (值越高越相似)")
        
        # 可视化
        if source_vis and target_vis:
            source_grid = make_grid(source_vis, nrow=vis_samples)
            target_grid = make_grid(target_vis, nrow=vis_samples)
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(source_grid.permute(1, 2, 0).cpu().numpy())
            plt.title("原始图像")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(target_grid.permute(1, 2, 0).cpu().numpy())
            plt.title("化妆后图像")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "samples_visualization.png"))
            plt.close()
            
            print(f"样本可视化已保存到 {os.path.join(OUTPUT_DIR, 'samples_visualization.png')}。")

# 执行代码
if __name__ == "__main__":
    # 加载面部掩码模型
    get_face_mask = load_farl_model()
    
    # 加载 DDA 模型
    pipeline, rdm, fgmm = load_dda_model()
    
    # 选择 FFHQ 图像 (4000 张)
    selected_images = select_ffhq_images(4000)
    
    # 数据增强
    amplify_data(selected_images, pipeline, rdm, fgmm, get_face_mask)
    
    # 评估增强数据
    evaluate_amplified_data(50)
    
    print("第3阶段：使用 DDA 进行数据增强完成")

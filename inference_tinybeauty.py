#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TinyBeauty 模型推理脚本
用于对单张图像应用训练好的 TinyBeauty 美妆模型
"""

import torch
import torchvision.transforms as T
import numpy as np
import cv2
import argparse
import os
from PIL import Image

# === 替换为你的 TinyBeauty 模型路径 ===
MODEL_PATH = "models/tinybeauty_best.pth"

# === TinyBeauty 模型结构定义（需与你训练代码一致） ===
class TinyBeautyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, stride=1, padding=1),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(16, 3, 3, stride=1, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def load_image(path, size=256):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor()
    ])
    return transform(img).unsqueeze(0)  # shape: (1, 3, H, W)

def save_image(tensor, path):
    img = tensor.squeeze(0).detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW to HWC
    img = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def beautify(image_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = TinyBeautyNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 加载图像
    x = load_image(image_path).to(device)

    # 推理
    with torch.no_grad():
        y = model(x)

    # 保存结果
    save_image(y, output_path)
    print(f"美妆图像已保存到: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='原始人脸图像路径')
    parser.add_argument('--output', type=str, required=True, help='输出图像路径')
    args = parser.parse_args()

    beautify(args.input, args.output)

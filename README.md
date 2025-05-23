# 移动环境中的高质量人脸美妆技术 (TinyBeauty)

![TinyBeauty 로고](images/tinybeauty_logo.svg)

## 项目概要

本项目旨在开发可用于移动环境的高质量人脸美妆技术。主要技术挑战如下:

1. **模型大小**: 适用于移动部署的小模型大小(<100KB)
2. **推理速度**: 移动设备上的实时性能(低于3ms)
3. **美妆质量**: 高画质(PSNR高于34dB)
4. **身份保持**: 保持面部特征
5. **高效训练**: 即使是有限的训练数据(5对)也可进行训练

为了解决这些问题，我们开发并实现了 Data Amplify Learning (DAL) 框架和 TinyBeauty 模型。

## 成果摘要

- **模型大小**: 81KB (达成目标：小于100KB)
- **推理延迟时间**: 2.18ms (以 iPhone 13 为基准，达成目标：低于3ms)
- **美妆质量**: 35.39dB PSNR (达成目标：高于34dB)
- **用户评价**: 相较于现有方法，在美妆质量与身份保持方面获得最高评价

## 技术贡献

1. **Data Amplify Learning**: 一种在初始数据有限的情况下生成高质量训练数据的新方法
2. **残差学习(Residual Learning)**: 模型仅学习图像之间的差异（残差），以保留细节
3. **眼线损失函数**: 利用 Sobel 滤波器的特殊损失函数强化精致的美妆细节
4. **轻量模型设计**: 针对移动端约束条件开发的高效模型架构

## 项目结构



```
.
├── 1_data_preprocessing/ # 数据种子选择与标注处理
├── 2_dda_training/ # Diffusion-based Data Amplifier (DDA) 训练
├── 3_data_amplification/ # 使用 DDA 进行数据增强
├── 4_tinybeauty_training/ # TinyBeauty 模型训练
├── 5_model_optimization/ # 模型优化与 CoreML 转换
├── 6_ablation_studies/ # 眼线损失与细节增强验证
├── 7_user_study/ # 用户研究与主观评价
├── images/ # 图像与可视化资料
└── models/ # 保存的模型文件
```

## 执行方法

关于各阶段代码的执行说明，请参考对应目录中的 README 文件。

### 1. 环境设置

```bash
# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装所需依赖包
pip install -r requirements.txt

### 2. 파이프라인 실행

```bash
# 각 단계별로 실행하거나
python run_pipeline.py --stage=all  # 전체 파이프라인 실행
```

## 参考文献

本项目基于以下研究：

- Stable Diffusion (Rombach et al., 2022)
- BeautyGAN (Li et al., 2018)
- FaRL: Face Representation Learning (Zheng et al., 2022)

## 许可证

MIT License

## 作者

- 研究与实现: JJshome

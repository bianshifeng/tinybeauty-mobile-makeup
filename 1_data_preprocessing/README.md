# 1단계: 데이터 시드 선택 및 주석 처리

在此阶段，准备用于化妆迁移模型训练的初始数据。

## 目的

- 从FFHQ数据集中选择5张适合化妆应用的面部图像
- 对选定的图像手动应用5种不同的化妆风格
- 生成原始-化妆图像对(pair)并进行质量管理

## 主要功能

- `select_seed_images()`: 从FFHQ数据集中选择5张面部图像
- `prepare_paired_data()`: 生成化妆图像与原始图像对
- `visualize_pairs()`: 可视化图像对以检查质量

## 使用方法

```bash
# 运行脚本
python data_preprocessing.py
```

## 输出结果

- `seed_data/`: 存储选定的原始图像和化妆图像的目录
- `seed_data/seed_image_{i}.png`: 选定的原始图像
- `seed_data/seed_image_{i}_makeup.png`: 应用化妆的图像
- `seed_data/seed_pairs_visualization.png`: 原始-化妆图像对的可视化

## 注意事项

- 实际实现中需要使用MEITU或类似软件进行化妆应用。
- 化妆风格应包括多种特性（如口红颜色、眼影、眼线等）。
- 原始图像应包含多样化的面部形状、性别、年龄等。

## 质量管理

- 目视检查注释化妆图像的准确性和一致性
- 确保化妆风格清晰且具有代表性
- 检查面部对齐状态（化妆前后面部位置是否一致）


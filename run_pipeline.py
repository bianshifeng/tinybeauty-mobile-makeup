#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TinyBeauty 移动美妆流水线运行脚本

本脚本用于执行适用于移动设备的人脸美妆模型研发流程。
共包含 7 个阶段，可依次运行，亦可选择某一特定阶段单独执行。
"""

import os
import sys
import argparse
import time
import importlib.util
from datetime import datetime

def import_module_from_file(module_name, file_path):
    """
    从文件路径动态导入模块
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_stage(stage_number, stage_name, module_path):
    """
    执行指定阶段的模块代码
    """
    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"阶段 {stage_number}: 正在执行 {stage_name}...")
    print(f"{'='*80}\n")
    
    try:
        file_name = os.path.basename(module_path)
        module_name = os.path.splitext(file_name)[0]
        module = import_module_from_file(module_name, module_path)

        # 如果模块有 __main__ 函数则执行
        if hasattr(module, '__main__'):
            module.__main__()

        elapsed_time = time.time() - start_time
        print(f"\n阶段 {stage_number}: {stage_name} 完成！（耗时: {elapsed_time:.2f} 秒）")
        return True

    except Exception as e:
        print(f"\n阶段 {stage_number}: {stage_name} 执行出错：")
        print(f"错误内容: {str(e)}")
        return False

def run_pipeline(start_stage=1, end_stage=7):
    """
    按照设定范围执行若干阶段
    """
    pipeline_start_time = time.time()

    # 各阶段定义（阶段编号, 名称, 模块路径）
    stages = [
        (1, "数据种子选择与标注处理", "1_data_preprocessing/data_preprocessing.py"),
        (2, "Diffusion 数据增强器（DDA）训练", "2_dda_training/dda_training.py"),
        (3, "基于 DDA 的数据增强生成", "3_data_amplification/data_amplification.py"),
        (4, "TinyBeauty 模型训练", "4_tinybeauty_training/tinybeauty_training.py"),
        (5, "模型优化及 CoreML 导出", "5_model_optimization/model_optimization.py"),
        (6, "眼妆损失/细节增强效果验证（消融实验）", "6_ablation_studies/ablation_studies.py"),
        (7, "用户调研与主观视觉评价", "7_user_study/user_study.py")
    ]

    # 根据输入选择阶段
    stages_to_run = [s for s in stages if s[0] >= start_stage and s[0] <= end_stage]

    # 执行前输出阶段清单
    print(f"\n开始执行 TinyBeauty 流水线（阶段 {start_stage} ~ {end_stage}）")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n将执行以下阶段：")
    for stage_number, stage_name, _ in stages_to_run:
        print(f"  - 阶段 {stage_number}: {stage_name}")
    print("\n")

    # 执行所有阶段
    success_count = 0
    for stage_number, stage_name, module_path in stages_to_run:
        success = run_stage(stage_number, stage_name, module_path)
        if success:
            success_count += 1

    # 执行结果总结
    pipeline_elapsed_time = time.time() - pipeline_start_time
    print(f"\n{'='*80}")
    print("流水线执行完成！")
    print(f"{'='*80}")
    print(f"共 {len(stages_to_run)} 个阶段，成功执行 {success_count} 个")
    print(f"总耗时: {pipeline_elapsed_time:.2f} 秒")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyBeauty 美妆开发流水线执行器")
    parser.add_argument('--stage', type=str, default='all',
                        help='指定执行阶段（如: "1", "2-5", "all"）')
    args = parser.parse_args()

    # 解析参数
    if args.stage == 'all':
        start_stage, end_stage = 1, 7
    elif '-' in args.stage:
        start_stage, end_stage = map(int, args.stage.split('-'))
    else:
        start_stage = end_stage = int(args.stage)

    # 执行主流程
    run_pipeline(start_stage, end_stage)

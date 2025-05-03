#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TinyBeauty 모바일 메이크업 파이프라인 실행 스크립트

이 스크립트는 모바일 환경에서의 고품질 얼굴 메이크업 적용 기술 개발 파이프라인을 
단계별로 실행합니다. 전체 7단계의 과정을 순차적으로 실행하거나 특정 단계만 실행할 수 있습니다.
"""

import os
import sys
import argparse
import time
import importlib.util
from datetime import datetime

def import_module_from_file(module_name, file_path):
    """
    파일 경로에서 모듈을 동적으로 임포트합니다.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_stage(stage_number, stage_name, module_path):
    """
    특정 단계의 코드를 실행합니다.
    """
    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"단계 {stage_number}: {stage_name} 실행 중...")
    print(f"{'='*80}\n")
    
    try:
        # 모듈 경로에서 파일 이름 추출
        file_name = os.path.basename(module_path)
        module_name = os.path.splitext(file_name)[0]
        
        # 모듈 임포트 및 실행
        module = import_module_from_file(module_name, module_path)
        
        # 모듈 메인 함수 실행
        if hasattr(module, '__main__'):
            module.__main__()
        
        elapsed_time = time.time() - start_time
        print(f"\n단계 {stage_number}: {stage_name} 완료! (소요 시간: {elapsed_time:.2f}초)")
        return True
        
    except Exception as e:
        print(f"\n단계 {stage_number}: {stage_name} 실행 중 오류 발생:")
        print(f"오류 내용: {str(e)}")
        return False

def run_pipeline(start_stage=1, end_stage=7):
    """
    지정된 범위의 파이프라인 단계를 실행합니다.
    """
    pipeline_start_time = time.time()
    
    # 각 단계별 설정
    stages = [
        (1, "데이터 시드 선택 및 주석 처리", "1_data_preprocessing/data_preprocessing.py"),
        (2, "Diffusion-based Data Amplifier (DDA) 훈련", "2_dda_training/dda_training.py"),
        (3, "DDA를 이용한 데이터 증폭", "3_data_amplification/data_amplification.py"),
        (4, "TinyBeauty 모델 훈련", "4_tinybeauty_training/tinybeauty_training.py"),
        (5, "모델 최적화 및 CoreML 변환", "5_model_optimization/model_optimization.py"),
        (6, "아이라이너 손실 및 디테일 강화 검증", "6_ablation_studies/ablation_studies.py"),
        (7, "사용자 연구 및 주관적 평가", "7_user_study/user_study.py")
    ]
    
    # 실행 범위 제한
    stages_to_run = [s for s in stages if s[0] >= start_stage and s[0] <= end_stage]
    
    # 실행 전 스테이지 정보 출력
    print(f"\nTinyBeauty 파이프라인 실행 (단계 {start_stage} ~ {end_stage})")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n실행할 단계:")
    for stage_number, stage_name, _ in stages_to_run:
        print(f"  - 단계 {stage_number}: {stage_name}")
    print("\n")
    
    # 각 단계 실행
    success_count = 0
    for stage_number, stage_name, module_path in stages_to_run:
        success = run_stage(stage_number, stage_name, module_path)
        if success:
            success_count += 1
    
    # 결과 요약
    pipeline_elapsed_time = time.time() - pipeline_start_time
    print(f"\n{'='*80}")
    print(f"파이프라인 실행 완료!")
    print(f"{'='*80}")
    print(f"총 {len(stages_to_run)}개 단계 중 {success_count}개 성공")
    print(f"총 소요 시간: {pipeline_elapsed_time:.2f}초")
    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyBeauty 모바일 메이크업 파이프라인 실행")
    parser.add_argument('--stage', type=str, default='all',
                        help='실행할 단계 (예: "1", "2-5", "all")')
    args = parser.parse_args()
    
    # 인자 파싱
    if args.stage == 'all':
        start_stage, end_stage = 1, 7
    elif '-' in args.stage:
        start_stage, end_stage = map(int, args.stage.split('-'))
    else:
        start_stage = end_stage = int(args.stage)
    
    # 파이프라인 실행
    run_pipeline(start_stage, end_stage)

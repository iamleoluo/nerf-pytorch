#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append('data_validation')
from validators.data_validator import DataValidator

def test_validation_details():
    PROJECT_ROOT = Path('.')
    COLMAP_OUTPUT = PROJECT_ROOT / 'camper_nerf' / 'colmap_output'
    NERF_DATA = PROJECT_ROOT / 'camper_nerf' / 'nerf_data'
    RAW_IMAGES = PROJECT_ROOT / 'camper_nerf' / 'images'

    print("=== 詳細驗證測試 ===")
    
    try:
        validator = DataValidator(COLMAP_OUTPUT, NERF_DATA, RAW_IMAGES)
        print("✅ DataValidator初始化成功")
        
        # 檢查數據載入情況
        print(f"\n📊 數據載入情況:")
        print(f"  COLMAP相機: {len(validator.colmap_cameras)}")
        print(f"  COLMAP圖像: {len(validator.colmap_images)}")
        print(f"  NeRF相機: {len(validator.nerf_cameras)}")
        print(f"  原始圖像: {len(validator.raw_images)}")
        
        # 測試每個驗證方法
        print(f"\n🔍 測試各個驗證方法:")
        
        print("1. 測試 validate_camera_consistency()...")
        camera_results = validator.validate_camera_consistency()
        print(f"   結果數量: {len(camera_results)}")
        if camera_results:
            print(f"   第一個結果: {camera_results[0]}")
        
        print("2. 測試 validate_image_quality()...")
        quality_results = validator.validate_image_quality()
        print(f"   結果數量: {len(quality_results)}")
        if quality_results:
            print(f"   第一個結果: {quality_results[0]}")
        
        print("3. 測試 validate_data_completeness()...")
        completeness_results = validator.validate_data_completeness()
        print(f"   結果數量: {len(completeness_results)}")
        if completeness_results:
            print(f"   第一個結果: {completeness_results[0]}")
            
        # 檢查nerf_cameras的結構
        print(f"\n🔍 NeRF相機數據結構:")
        if validator.nerf_cameras:
            first_key = list(validator.nerf_cameras.keys())[0]
            print(f"  第一個鍵: {first_key}")
            print(f"  第一個值: {validator.nerf_cameras[first_key]}")
            
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_validation_details() 
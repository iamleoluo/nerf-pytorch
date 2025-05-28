#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append('data_validation')
from validators.data_validator import DataValidator

def test_validate_all():
    PROJECT_ROOT = Path('.')
    COLMAP_OUTPUT = PROJECT_ROOT / 'camper_nerf' / 'colmap_output'
    NERF_DATA = PROJECT_ROOT / 'camper_nerf' / 'nerf_data'
    RAW_IMAGES = PROJECT_ROOT / 'camper_nerf' / 'images'

    print("=== 測試DataValidator.validate_all() ===")
    
    try:
        validator = DataValidator(COLMAP_OUTPUT, NERF_DATA, RAW_IMAGES)
        print("✅ DataValidator初始化成功")
        
        print("正在執行validate_all()...")
        results = validator.validate_all()
        print(f"✅ validate_all()執行成功，返回類型: {type(results)}")
        
        if isinstance(results, dict):
            print(f"結果字典鍵: {list(results.keys())}")
            for category, category_results in results.items():
                print(f"  {category}: {len(category_results)} 個結果")
                if category_results:
                    print(f"    第一個結果: {category_results[0]}")
        else:
            print(f"結果: {results}")
            
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_validate_all() 
#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append('data_validation')
from validators.data_validator import DataValidator

def debug_colmap_structure():
    PROJECT_ROOT = Path('.')
    COLMAP_OUTPUT = PROJECT_ROOT / 'camper_nerf' / 'colmap_output'
    NERF_DATA = PROJECT_ROOT / 'camper_nerf' / 'nerf_data'
    RAW_IMAGES = PROJECT_ROOT / 'camper_nerf' / 'images'

    validator = DataValidator(COLMAP_OUTPUT, NERF_DATA, RAW_IMAGES)
    
    print("=== COLMAP圖像數據結構調試 ===")
    print(f"COLMAP圖像總數: {len(validator.colmap_images)}")
    
    if validator.colmap_images:
        # 顯示前2個圖像的數據結構
        for i, (key, value) in enumerate(list(validator.colmap_images.items())[:2]):
            print(f"\n圖像 {i+1}: {key}")
            print(f"數據結構: {type(value)}")
            if isinstance(value, dict):
                print("字典鍵:")
                for k in value.keys():
                    print(f"  - {k}: {type(value[k])}")
            else:
                print(f"值: {value}")
    else:
        print("沒有COLMAP圖像數據")

if __name__ == "__main__":
    debug_colmap_structure() 
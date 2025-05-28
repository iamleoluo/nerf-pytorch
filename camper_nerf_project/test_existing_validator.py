#!/usr/bin/env python3
"""
測試修復後的現有data_validation系統
"""

import sys
import os
from pathlib import Path

# 添加data_validation目錄到Python路徑
sys.path.append(str(Path(__file__).parent / "data_validation"))

from validators.data_validator import DataValidator

def test_existing_validator():
    # 配置路徑
    PROJECT_ROOT = Path(__file__).parent
    COLMAP_OUTPUT = PROJECT_ROOT / "camper_nerf" / "colmap_output"
    NERF_DATA = PROJECT_ROOT / "camper_nerf" / "nerf_data"
    RAW_IMAGES = PROJECT_ROOT / "camper_nerf" / "images"
    
    print("=== 測試現有data_validation系統 ===")
    print(f"COLMAP輸出目錄: {COLMAP_OUTPUT}")
    print(f"NeRF數據目錄: {NERF_DATA}")
    print(f"原始圖像目錄: {RAW_IMAGES}")
    print()
    
    # 檢查路徑是否存在
    for path, name in [
        (COLMAP_OUTPUT, "COLMAP輸出目錄"),
        (NERF_DATA, "NeRF數據目錄"),
        (RAW_IMAGES, "原始圖像目錄")
    ]:
        if path.exists():
            print(f"✅ {name}存在: {path}")
        else:
            print(f"❌ {name}不存在: {path}")
    print()
    
    # 創建驗證器
    try:
        print("正在初始化DataValidator...")
        validator = DataValidator(COLMAP_OUTPUT, NERF_DATA, RAW_IMAGES)
        print("✅ DataValidator初始化成功")
        print()
        
        # 顯示載入的數據統計
        print("=== 數據載入統計 ===")
        print(f"📷 COLMAP相機內參: {len(validator.colmap_cameras)}")
        print(f"📸 COLMAP圖像姿態: {len(validator.colmap_images)}")
        print(f"🎯 NeRF相機: {len(validator.nerf_cameras)}")
        print(f"🖼️  原始圖像: {len(validator.raw_images)}")
        print()
        
        # 顯示前5個COLMAP圖像
        if validator.colmap_images:
            print("前5個COLMAP圖像:")
            for i, (name, data) in enumerate(list(validator.colmap_images.items())[:5]):
                print(f"   {i+1}. {name} (ID: {data['image_id']})")
        print()
        
        # 執行驗證
        print("正在執行數據驗證...")
        validation_results = validator.validate_data()
        
        print("=== 驗證結果 ===")
        for category, result in validation_results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"{status_icon} {category}: {result['status']}")
            if result['issues']:
                for issue in result['issues']:
                    print(f"   - {issue}")
        print()
        
        # 檢查數據一致性
        nerf_count = len(validator.nerf_cameras)
        colmap_count = len(validator.colmap_images)
        raw_count = len(validator.raw_images)
        
        print("=== 數據一致性檢查 ===")
        print(f"NeRF相機數量: {nerf_count}")
        print(f"COLMAP圖像數量: {colmap_count}")
        print(f"原始圖像數量: {raw_count}")
        
        if nerf_count == colmap_count == raw_count:
            print("✅ 所有數據數量完美匹配！")
        else:
            print("⚠️  數據數量不匹配，需要進一步檢查")
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_existing_validator() 
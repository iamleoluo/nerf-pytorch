#!/usr/bin/env python3
"""
測試修復後的data_validation系統
"""

import sys
import os
from pathlib import Path

# 添加data_validation目錄到Python路徑
sys.path.append(str(Path(__file__).parent / "data_validation"))

def test_system():
    try:
        from validators.data_validator import DataValidator
        
        # 配置路徑
        PROJECT_ROOT = Path(__file__).parent
        COLMAP_OUTPUT = PROJECT_ROOT / "camper_nerf" / "colmap_output"
        NERF_DATA = PROJECT_ROOT / "camper_nerf" / "nerf_data"
        RAW_IMAGES = PROJECT_ROOT / "camper_nerf" / "images"
        
        print("🔧 測試修復後的data_validation系統")
        print("=" * 50)
        
        # 檢查路徑
        print("📁 檢查路徑:")
        for path, name in [
            (COLMAP_OUTPUT, "COLMAP輸出"),
            (NERF_DATA, "NeRF數據"),
            (RAW_IMAGES, "原始圖像")
        ]:
            status = "✅" if path.exists() else "❌"
            print(f"   {status} {name}: {path}")
        
        print("\n🔄 初始化DataValidator...")
        validator = DataValidator(COLMAP_OUTPUT, NERF_DATA, RAW_IMAGES)
        
        print("\n📊 數據載入結果:")
        print(f"   📷 COLMAP相機內參: {len(validator.colmap_cameras)}")
        print(f"   📸 COLMAP圖像姿態: {len(validator.colmap_images)}")
        print(f"   🎯 NeRF相機: {len(validator.nerf_cameras)}")
        print(f"   🖼️  原始圖像: {len(validator.raw_images)}")
        
        # 檢查數據一致性
        nerf_count = len(validator.nerf_cameras)
        colmap_count = len(validator.colmap_images)
        raw_count = len(validator.raw_images)
        
        print("\n🔍 數據一致性檢查:")
        print(f"   NeRF相機數量: {nerf_count}")
        print(f"   COLMAP圖像數量: {colmap_count}")
        print(f"   原始圖像數量: {raw_count}")
        
        if nerf_count == colmap_count == raw_count:
            print("   ✅ 所有數據數量完美匹配！")
            success = True
        else:
            print("   ⚠️  數據數量不匹配")
            success = False
        
        # 顯示一些樣本數據
        if validator.colmap_images:
            print("\n📋 COLMAP圖像樣本:")
            for i, (name, data) in enumerate(list(validator.colmap_images.items())[:3]):
                print(f"   {i+1}. {name} (ID: {data['image_id']})")
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 系統修復成功！所有數據正確載入並匹配。")
        else:
            print("⚠️  系統部分修復，但仍有數據不匹配問題。")
        
        return success
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_system() 
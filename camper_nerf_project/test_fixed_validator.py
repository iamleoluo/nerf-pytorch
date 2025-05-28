#!/usr/bin/env python3
"""
測試修復後的DataValidator
"""

import sys
import os
sys.path.append('data_validation_system/backend')

from data_validator import DataValidator

def test_fixed_validator():
    # 配置路徑
    BASE_DIR = "/home/leoluo/文件/GitHub/nerf-pytorch/camper_nerf_project/camper_nerf"
    COLMAP_DIR = os.path.join(BASE_DIR, "colmap_output")
    IMAGES_DIR = os.path.join(BASE_DIR, "images")
    NERF_DATA_PATH = os.path.join(BASE_DIR, "nerf_data", "transforms.json")
    
    print("=== 測試修復後的DataValidator ===")
    print(f"COLMAP目錄: {COLMAP_DIR}")
    print(f"圖像目錄: {IMAGES_DIR}")
    print(f"NeRF數據: {NERF_DATA_PATH}")
    print()
    
    # 創建驗證器
    validator = DataValidator(COLMAP_DIR, IMAGES_DIR, NERF_DATA_PATH)
    
    # 獲取驗證摘要
    print("正在執行驗證...")
    results = validator.get_validation_summary()
    
    print("=== 驗證結果 ===")
    
    # 相機數量驗證
    camera_result = results['camera_count']
    print(f"📷 相機數量驗證:")
    print(f"   NeRF相機: {camera_result['nerf_cameras']}")
    print(f"   COLMAP相機: {camera_result['colmap_cameras']}")
    print(f"   匹配狀態: {'✅ 匹配' if camera_result['match'] else '❌ 不匹配'}")
    print()
    
    # 圖像文件驗證
    image_result = results['image_files']
    print(f"🖼️  圖像文件驗證:")
    print(f"   缺失文件: {image_result['total_missing']}")
    print(f"   存在文件: {len(image_result['existing_files'])}")
    if image_result['missing_files']:
        print(f"   缺失的文件: {image_result['missing_files'][:5]}...")
    print()
    
    # 圖像質量驗證
    quality_result = results['image_quality']
    print(f"🔍 圖像質量驗證:")
    print(f"   質量問題: {quality_result['total_issues']}")
    if quality_result['issues']:
        print(f"   問題詳情: {quality_result['issues'][:3]}...")
    print()
    
    # 總體狀態
    print(f"📊 總體狀態: {'✅ 成功' if results['overall_status'] == 'success' else '⚠️ 警告'}")
    
    # 詳細的COLMAP數據信息
    print("\n=== COLMAP數據詳情 ===")
    print(f"載入的COLMAP圖像數量: {len(validator.colmap_images)}")
    if validator.colmap_images:
        print("前5個COLMAP圖像:")
        for i, (image_id, image_data) in enumerate(list(validator.colmap_images.items())[:5]):
            print(f"   {image_id}: {image_data['name']}")

if __name__ == "__main__":
    test_fixed_validator() 
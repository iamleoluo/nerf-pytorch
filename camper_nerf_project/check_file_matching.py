#!/usr/bin/env python3
"""
檢查NeRF、COLMAP和原始圖像文件之間的匹配情況
"""

import os
import json

def check_file_matching():
    # 路徑設定
    images_dir = "/home/leoluo/文件/GitHub/nerf-pytorch/camper_nerf_project/camper_nerf/images"
    transforms_path = "/home/leoluo/文件/GitHub/nerf-pytorch/camper_nerf_project/camper_nerf/nerf_data/transforms.json"
    colmap_images_txt = "/home/leoluo/文件/GitHub/nerf-pytorch/camper_nerf_project/camper_nerf/colmap_output/sparse/0/images.txt"
    
    print("=== 文件匹配檢查 ===")
    print(f"原始圖像目錄: {images_dir}")
    print(f"NeRF transforms.json: {transforms_path}")
    print(f"COLMAP images.txt: {colmap_images_txt}")
    print()
    
    # 1. 獲取原始圖像目錄中的所有圖像文件名（去除擴展名）
    if not os.path.exists(images_dir):
        print(f"錯誤：原始圖像目錄不存在: {images_dir}")
        return
    
    raw_files = os.listdir(images_dir)
    raw_images = set()
    for f in raw_files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(f)[0]
            raw_images.add(base_name)
    
    print(f"原始圖像目錄中找到 {len(raw_images)} 個圖像文件")
    print(f"前5個文件名: {list(raw_images)[:5]}")
    print()
    
    # 2. 獲取transforms.json中的所有file_path（去除目錄和擴展名）
    if not os.path.exists(transforms_path):
        print(f"錯誤：transforms.json不存在: {transforms_path}")
        return
    
    try:
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        
        nerf_files = set()
        for frame in transforms['frames']:
            file_path = frame['file_path']
            # 去除目錄和擴展名
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            nerf_files.add(base_name)
        
        print(f"transforms.json中找到 {len(nerf_files)} 個幀")
        print(f"前5個file_path: {[frame['file_path'] for frame in transforms['frames'][:5]]}")
        print(f"前5個基礎文件名: {list(nerf_files)[:5]}")
        print()
    except Exception as e:
        print(f"錯誤：無法讀取transforms.json: {e}")
        return
    
    # 3. 獲取COLMAP images.txt中的所有圖像名稱（去除擴展名）
    if not os.path.exists(colmap_images_txt):
        print(f"錯誤：COLMAP images.txt不存在: {colmap_images_txt}")
        return
    
    try:
        colmap_files = set()
        with open(colmap_images_txt, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 10:  # 圖像行應該有至少10個部分
                    image_name = parts[9]  # 最後一個部分是圖像名稱
                    base_name = os.path.splitext(os.path.basename(image_name))[0]
                    colmap_files.add(base_name)
        
        print(f"COLMAP images.txt中找到 {len(colmap_files)} 個圖像")
        print(f"前5個圖像名: {list(colmap_files)[:5]}")
        print()
    except Exception as e:
        print(f"錯誤：無法讀取COLMAP images.txt: {e}")
        return
    
    # 4. 進行匹配分析
    print("=== 匹配分析結果 ===")
    
    # NeRF中缺失的圖像（在transforms.json中但不在原始圖像目錄中）
    nerf_missing_in_raw = nerf_files - raw_images
    if nerf_missing_in_raw:
        print(f"❌ NeRF中的 {len(nerf_missing_in_raw)} 個文件在原始圖像目錄中缺失:")
        for f in sorted(nerf_missing_in_raw):
            print(f"   - {f}")
    else:
        print("✅ NeRF中的所有文件都在原始圖像目錄中找到")
    print()
    
    # NeRF中缺失的COLMAP圖像
    nerf_missing_in_colmap = nerf_files - colmap_files
    if nerf_missing_in_colmap:
        print(f"❌ NeRF中的 {len(nerf_missing_in_colmap)} 個文件在COLMAP中缺失:")
        for f in sorted(nerf_missing_in_colmap):
            print(f"   - {f}")
    else:
        print("✅ NeRF中的所有文件都在COLMAP中找到")
    print()
    
    # COLMAP中缺失的圖像
    colmap_missing_in_raw = colmap_files - raw_images
    if colmap_missing_in_raw:
        print(f"❌ COLMAP中的 {len(colmap_missing_in_raw)} 個文件在原始圖像目錄中缺失:")
        for f in sorted(colmap_missing_in_raw):
            print(f"   - {f}")
    else:
        print("✅ COLMAP中的所有文件都在原始圖像目錄中找到")
    print()
    
    # COLMAP中缺失的NeRF文件
    colmap_missing_in_nerf = colmap_files - nerf_files
    if colmap_missing_in_nerf:
        print(f"❌ COLMAP中的 {len(colmap_missing_in_nerf)} 個文件在NeRF中缺失:")
        for f in sorted(colmap_missing_in_nerf):
            print(f"   - {f}")
    else:
        print("✅ COLMAP中的所有文件都在NeRF中找到")
    print()
    
    # 未使用的原始圖像
    unused_raw_images = raw_images - (nerf_files | colmap_files)
    if unused_raw_images:
        print(f"⚠️  原始圖像目錄中有 {len(unused_raw_images)} 個文件未被NeRF或COLMAP使用:")
        for f in sorted(unused_raw_images):
            print(f"   - {f}")
    else:
        print("✅ 原始圖像目錄中的所有文件都被使用")
    print()
    
    # 總結
    print("=== 總結 ===")
    print(f"原始圖像: {len(raw_images)} 個文件")
    print(f"NeRF幀: {len(nerf_files)} 個文件")
    print(f"COLMAP圖像: {len(colmap_files)} 個文件")
    
    if len(nerf_files) == len(colmap_files) == len(raw_images) and not (nerf_missing_in_raw or nerf_missing_in_colmap or colmap_missing_in_raw or colmap_missing_in_nerf):
        print("✅ 所有文件完美匹配！")
    else:
        print("❌ 存在文件不匹配問題，需要修復")

if __name__ == "__main__":
    check_file_matching() 
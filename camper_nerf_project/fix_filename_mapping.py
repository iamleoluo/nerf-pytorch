#!/usr/bin/env python3
"""
修復COLMAP文件名映射問題
將實際圖片文件重命名為COLMAP記錄的格式
"""

import os
import shutil
from pathlib import Path
import read_write_model as rwm

def fix_filename_mapping(colmap_dir, images_dir, backup_dir=None):
    """
    修復文件名映射問題
    
    Args:
        colmap_dir: COLMAP輸出目錄 (包含sparse/0/)
        images_dir: 圖片目錄
        backup_dir: 備份目錄 (可選)
    """
    print("🔧 開始修復文件名映射...")
    
    # 讀取COLMAP記錄的圖片信息
    images_file = os.path.join(colmap_dir, "images.bin")
    if not os.path.exists(images_file):
        print("❌ 找不到COLMAP images.bin文件")
        return False
    
    images = rwm.read_images_binary(images_file)
    
    # 獲取實際圖片文件
    actual_files = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for file_path in Path(images_dir).iterdir():
        if file_path.suffix.lower() in image_extensions:
            actual_files.append(file_path.name)
    
    actual_files.sort()  # 按文件名排序
    
    # 獲取COLMAP記錄的文件名
    colmap_files = []
    for img_id, img_data in images.items():
        colmap_files.append((img_id, img_data.name))
    
    colmap_files.sort(key=lambda x: x[1])  # 按文件名排序
    
    print(f"📁 實際文件數量: {len(actual_files)}")
    print(f"📁 COLMAP記錄數量: {len(colmap_files)}")
    
    # 創建備份
    if backup_dir:
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        print(f"📦 創建備份到: {backup_path}")
        
        for actual_file in actual_files:
            src = Path(images_dir) / actual_file
            dst = backup_path / actual_file
            shutil.copy2(src, dst)
    
    # 策略1: 如果數量相等，按順序映射
    if len(actual_files) == len(colmap_files):
        print("✅ 文件數量匹配，按順序重命名")
        
        # 先重命名為臨時文件名，避免衝突
        temp_mappings = []
        for i, (actual_file, (img_id, colmap_file)) in enumerate(zip(actual_files, colmap_files)):
            temp_name = f"temp_{i:04d}.png"
            src = Path(images_dir) / actual_file
            temp_dst = Path(images_dir) / temp_name
            
            shutil.move(src, temp_dst)
            temp_mappings.append((temp_name, colmap_file))
            print(f"  臨時重命名: {actual_file} -> {temp_name}")
        
        # 再重命名為最終文件名
        for temp_name, colmap_file in temp_mappings:
            temp_src = Path(images_dir) / temp_name
            final_dst = Path(images_dir) / colmap_file
            
            shutil.move(temp_src, final_dst)
            print(f"  最終重命名: {temp_name} -> {colmap_file}")
        
        print("✅ 文件名映射修復完成")
        return True
    
    # 策略2: 數量不匹配，需要手動處理
    else:
        print("⚠️ 文件數量不匹配，需要手動處理")
        print("COLMAP記錄的文件:")
        for img_id, colmap_file in colmap_files[:10]:  # 只顯示前10個
            print(f"  {img_id}: {colmap_file}")
        if len(colmap_files) > 10:
            print(f"  ... 還有 {len(colmap_files) - 10} 個文件")
        
        print("\n實際文件:")
        for actual_file in actual_files[:10]:  # 只顯示前10個
            print(f"  {actual_file}")
        if len(actual_files) > 10:
            print(f"  ... 還有 {len(actual_files) - 10} 個文件")
        
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="修復COLMAP文件名映射問題")
    parser.add_argument("--colmap_dir", required=True, help="COLMAP輸出目錄 (包含sparse/0/)")
    parser.add_argument("--images_dir", required=True, help="圖片目錄")
    parser.add_argument("--backup_dir", help="備份目錄 (可選)")
    
    args = parser.parse_args()
    
    success = fix_filename_mapping(
        args.colmap_dir,
        args.images_dir,
        args.backup_dir
    )
    
    if not success:
        print("❌ 修復失敗")
        exit(1)
    else:
        print("🎉 修復成功！")

if __name__ == "__main__":
    main() 
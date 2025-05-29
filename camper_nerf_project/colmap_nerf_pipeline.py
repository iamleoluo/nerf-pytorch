#!/usr/bin/env python3
"""
COLMAP到NeRF自動化流水線
完整的從原始圖片到NeRF格式的自動化處理流程
"""

import os
import sys
import shutil
import subprocess
import argparse
import json
from pathlib import Path
import time
from datetime import datetime

# 導入配置
from config import ColmapNerfConfig, load_config_from_env

class ColmapNerfPipeline:
    def __init__(self, project_dir, raw_images_dir, verbose=True, config=None):
        """
        初始化流水線
        
        Args:
            project_dir: 項目工作目錄 (camper_nerf)
            raw_images_dir: 原始圖片目錄
            verbose: 是否顯示詳細信息
            config: 配置對象，如果為None則使用默認配置
        """
        self.project_dir = Path(project_dir)
        self.raw_images_dir = Path(raw_images_dir)
        self.verbose = verbose
        self.config = config or ColmapNerfConfig()
        
        # 使用配置創建目錄結構
        self.dirs = self.config.create_project_structure(self.project_dir)
        self.file_paths = self.config.get_file_paths(self.project_dir)
        
        self.log("✅ 目錄結構設置完成")
        
    def log(self, message, level="INFO"):
        """日誌輸出"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def step1_copy_images(self):
        """步驟1: 複製圖片到工作目錄"""
        self.log("🔄 步驟1: 開始複製圖片...")
        
        if not self.raw_images_dir.exists():
            raise FileNotFoundError(f"原始圖片目錄不存在: {self.raw_images_dir}")
        
        # 使用配置中的支持格式
        image_extensions = self.config.SUPPORTED_IMAGE_EXTENSIONS
        
        # 清空目標目錄
        if self.dirs['images_dir'].exists():
            shutil.rmtree(self.dirs['images_dir'])
        self.dirs['images_dir'].mkdir(parents=True)
        
        # 複製圖片
        copied_count = 0
        for image_file in self.raw_images_dir.iterdir():
            if image_file.suffix.lower() in image_extensions:
                dest_path = self.dirs['images_dir'] / image_file.name
                shutil.copy2(image_file, dest_path)
                copied_count += 1
                
        if copied_count == 0:
            raise ValueError(f"在 {self.raw_images_dir} 中沒有找到有效的圖片文件")
            
        self.log(f"✅ 步驟1完成: 複製了 {copied_count} 張圖片到工作目錄")
        return copied_count
    
    def step2_run_colmap(self):
        """步驟2: 執行COLMAP重建"""
        self.log("🔄 步驟2: 開始COLMAP處理...")
        
        # 獲取COLMAP環境變量
        colmap_env = self.config.get_colmap_env()
        
        # 2.1 特徵提取
        self.log("  2.1 執行特徵提取...")
        feature_cmd = self.config.get_colmap_feature_cmd(
            self.file_paths['database'],
            self.dirs['images_dir']
        )
        
        self.log(f"    執行命令: {' '.join(feature_cmd)}")
        self.log(f"    環境變量: QT_QPA_PLATFORM={colmap_env.get('QT_QPA_PLATFORM', 'default')}")
        
        result = subprocess.run(feature_cmd, capture_output=True, text=True, env=colmap_env)
        if result.returncode != 0:
            self.log(f"❌ 特徵提取失敗: {result.stderr}", "ERROR")
            # 如果是Qt相關錯誤，提供解決建議
            if "qt.qpa" in result.stderr.lower() or "xcb" in result.stderr.lower():
                self.log("💡 檢測到Qt顯示問題，嘗試禁用GPU或使用CPU模式", "INFO")
                self.log("   可以設置環境變量: export COLMAP_USE_GPU=0", "INFO")
            raise RuntimeError("COLMAP特徵提取失敗")
        
        # 2.2 特徵匹配
        self.log("  2.2 執行特徵匹配...")
        matcher_cmd = self.config.get_colmap_matcher_cmd(
            self.file_paths['database']
        )
        
        self.log(f"    執行命令: {' '.join(matcher_cmd)}")
        result = subprocess.run(matcher_cmd, capture_output=True, text=True, env=colmap_env)
        if result.returncode != 0:
            self.log(f"❌ 特徵匹配失敗: {result.stderr}", "ERROR")
            raise RuntimeError("COLMAP特徵匹配失敗")
        
        # 2.3 稀疏重建
        self.log("  2.3 執行稀疏重建...")
        mapper_cmd = self.config.get_colmap_mapper_cmd(
            self.file_paths['database'],
            self.dirs['images_dir'],
            self.dirs['sparse_dir'].parent
        )
        
        self.log(f"    執行命令: {' '.join(mapper_cmd)}")
        result = subprocess.run(mapper_cmd, capture_output=True, text=True, env=colmap_env)
        if result.returncode != 0:
            self.log(f"❌ 稀疏重建失敗: {result.stderr}", "ERROR")
            raise RuntimeError("COLMAP稀疏重建失敗")
        
        # 檢查輸出
        if not self.file_paths['cameras_bin'].exists():
            raise RuntimeError("COLMAP重建失敗，沒有生成cameras.bin")
            
        self.log("✅ 步驟2完成: COLMAP重建成功")
    
    def step3_convert_to_nerf(self):
        """步驟3: 轉換為NeRF格式"""
        self.log("🔄 步驟3: 開始轉換為NeRF格式...")
        
        # 導入轉換模塊
        sys.path.append(str(Path(__file__).parent))
        from colmap2nerf_fixed import colmap_to_nerf_fixed
        
        success = colmap_to_nerf_fixed(
            str(self.dirs['colmap_output_dir']),
            str(self.dirs['images_dir']),
            str(self.file_paths['transforms'])
        )
        
        if not success:
            raise RuntimeError("NeRF格式轉換失敗")
            
        self.log("✅ 步驟3完成: NeRF格式轉換成功")
        return self.file_paths['transforms']
    
    def step4_validate_data(self):
        """步驟4: 數據驗證建議"""
        self.log("🔄 步驟4: 數據驗證建議...")
        
        if not self.file_paths['transforms'].exists():
            self.log("❌ transforms.json 文件不存在", "ERROR")
            return False
        
        # 讀取並分析transforms.json
        with open(self.file_paths['transforms'], 'r') as f:
            data = json.load(f)
        
        frame_count = len(data.get('frames', []))
        camera_angle_x = data.get('camera_angle_x', 0)
        
        self.log(f"  📊 數據統計:")
        self.log(f"    - 圖片數量: {frame_count}")
        self.log(f"    - 相機視野角: {camera_angle_x:.4f} 弧度 ({camera_angle_x * 180 / 3.14159:.2f}°)")
        
        # 使用配置中的驗證參數
        validation_config = self.config.VALIDATION_CONFIG
        min_frames = validation_config['pose_quality']['min_frame_count']
        max_frames = validation_config['pose_quality']['max_frame_count']
        
        if frame_count < min_frames:
            self.log(f"    ⚠️ 圖片數量過少 (< {min_frames})", "WARNING")
        elif frame_count > max_frames:
            self.log(f"    ⚠️ 圖片數量過多 (> {max_frames})", "WARNING")
        else:
            self.log(f"    ✅ 圖片數量合適")
        
        # 驗證建議
        self.log("  🔍 建議進行以下驗證:")
        self.log("    1. 運行數據驗證工具檢查相機姿態")
        self.log("    2. 可視化相機軌跡確認合理性")
        self.log("    3. 檢查圖片質量和覆蓋範圍")
        
        validation_cmd = f"cd {Path(__file__).parent / 'data_validation'} && python app.py"
        self.log(f"  💡 運行驗證工具: {validation_cmd}")
        
        self.log("✅ 步驟4完成: 請手動進行數據驗證")
        return True
    
    def run_full_pipeline(self):
        """運行完整流水線"""
        start_time = time.time()
        
        self.log("🚀 開始COLMAP到NeRF完整流水線")
        self.log(f"📁 項目目錄: {self.project_dir}")
        self.log(f"📷 原始圖片: {self.raw_images_dir}")
        
        try:
            # 步驟1: 複製圖片
            image_count = self.step1_copy_images()
            
            # 步驟2: COLMAP處理
            self.step2_run_colmap()
            
            # 步驟3: 轉換格式
            transforms_file = self.step3_convert_to_nerf()
            
            # 步驟4: 驗證建議
            self.step4_validate_data()
            
            elapsed_time = time.time() - start_time
            
            self.log("🎉 流水線執行完成!")
            self.log(f"⏱️  總耗時: {elapsed_time:.2f} 秒")
            self.log(f"📄 輸出文件: {transforms_file}")
            self.log("📋 下一步: 請運行數據驗證工具確認結果")
            
            return True
            
        except Exception as e:
            self.log(f"❌ 流水線執行失敗: {str(e)}", "ERROR")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="COLMAP到NeRF自動化流水線",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python colmap_nerf_pipeline.py --raw_images raw_images --project camper_nerf
  python colmap_nerf_pipeline.py --raw_images /path/to/images --project /path/to/project --verbose
  
環境變量配置:
  COLMAP_USE_GPU=0          # 禁用GPU加速
  COLMAP_MAX_IMAGE_SIZE=2048 # 設置最大圖片尺寸
  COLMAP_MAX_FEATURES=4096   # 設置最大特徵點數
        """
    )
    
    parser.add_argument(
        "--raw_images", 
        required=True,
        help="原始圖片目錄路徑"
    )
    
    parser.add_argument(
        "--project", 
        required=True,
        help="項目工作目錄路徑 (將作為COLMAP工作區)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="顯示詳細執行信息"
    )
    
    parser.add_argument(
        "--skip-colmap", 
        action="store_true",
        help="跳過COLMAP處理 (僅轉換現有結果)"
    )
    
    parser.add_argument(
        "--config-from-env", 
        action="store_true",
        help="從環境變量加載配置"
    )
    
    args = parser.parse_args()
    
    # 加載配置
    if args.config_from_env:
        config = load_config_from_env()
        print("📋 使用環境變量配置")
    else:
        config = load_config_from_env()  # 總是加載環境變量配置
        print("📋 使用默認配置 (包含環境變量)")
    
    # 創建流水線實例
    pipeline = ColmapNerfPipeline(
        project_dir=args.project,
        raw_images_dir=args.raw_images,
        verbose=args.verbose,
        config=config
    )
    
    if args.skip_colmap:
        # 僅執行轉換步驟
        try:
            pipeline.step3_convert_to_nerf()
            pipeline.step4_validate_data()
        except Exception as e:
            print(f"❌ 轉換失敗: {e}")
            sys.exit(1)
    else:
        # 執行完整流水線
        success = pipeline.run_full_pipeline()
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main() 
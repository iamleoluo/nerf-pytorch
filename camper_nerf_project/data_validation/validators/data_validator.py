import os
import json
import sqlite3
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
from typing import List, Dict, Any, Tuple
import logging
import re
import sys

# 添加read_write_model模塊的路徑
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import read_write_model as rwm
except ImportError:
    print("警告：無法導入read_write_model，將使用備用方法")
    rwm = None

class DataValidator:
    def __init__(self, colmap_output_path, nerf_data_path, raw_images_path):
        """初始化數據驗證器
        
        Args:
            colmap_output_path: COLMAP 輸出目錄路徑
            nerf_data_path: NeRF 數據目錄路徑
            raw_images_path: 原始圖像目錄路徑
        """
        self.colmap_output_path = Path(colmap_output_path)
        self.nerf_data_path = Path(nerf_data_path)
        self.raw_images_path = Path(raw_images_path)
        self.logger = logging.getLogger(__name__)
        
        # 初始化數據
        self.colmap_cameras = {}
        self.colmap_images = {}
        self.nerf_cameras = {}
        self.raw_images = {}
        
        # 加載數據
        self._load_colmap_data()
        self._load_nerf_data()
        self._load_raw_images()
        
    def _load_colmap_data(self):
        """使用read_write_model載入COLMAP數據"""
        try:
            # 嘗試從sparse/0目錄載入
            sparse_dir = self.colmap_output_path / "sparse" / "0"
            if not sparse_dir.exists():
                sparse_dir = self.colmap_output_path / "sparse"
            
            cameras_file = sparse_dir / "cameras.bin"
            images_file = sparse_dir / "images.bin"
            
            if not cameras_file.exists() or not images_file.exists():
                self.logger.error(f"找不到COLMAP二進制文件在: {sparse_dir}")
                return
            
            if rwm is None:
                self.logger.error("read_write_model模塊不可用，無法讀取COLMAP數據")
                return
            
            # 讀取相機內參
            self.colmap_cameras = rwm.read_cameras_binary(str(cameras_file))
            self.logger.info(f"載入 {len(self.colmap_cameras)} 個COLMAP相機內參")
            
            # 讀取圖像姿態
            colmap_images_raw = rwm.read_images_binary(str(images_file))
            
            # 獲取實際存在的圖像文件名
            actual_images = set()
            if self.raw_images_path.exists():
                for f in self.raw_images_path.iterdir():
                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        base_name = f.stem  # 文件名不含擴展名
                        actual_images.add(base_name)
            
            self.logger.info(f"找到 {len(actual_images)} 個實際圖像文件")
            
            # 過濾只保留實際存在的圖像
            loaded_count = 0
            skipped_count = 0
            
            for img_id, img_data in colmap_images_raw.items():
                # 檢查圖像是否實際存在
                base_name = Path(img_data.name).stem
                if base_name in actual_images:
                    self.colmap_images[img_data.name] = {
                        'image_id': img_id,
                        'qvec': img_data.qvec,  # [qw, qx, qy, qz]
                        'tvec': img_data.tvec,  # [tx, ty, tz]
                        'camera_id': img_data.camera_id,
                        'name': img_data.name
                    }
                    loaded_count += 1
                else:
                    skipped_count += 1
                    self.logger.debug(f"跳過不存在的圖像: {img_data.name} (基礎名: {base_name})")
            
            self.logger.info(f"載入 {loaded_count} 個COLMAP圖像姿態")
            if skipped_count > 0:
                self.logger.info(f"跳過 {skipped_count} 個不存在的圖像")
                
        except Exception as e:
            self.logger.error(f"載入COLMAP數據時出錯: {str(e)}")
            import traceback
            traceback.print_exc()

    def _load_nerf_data(self):
        """加載 NeRF 數據"""
        try:
            transforms_path = self.nerf_data_path / "transforms.json"
            if not transforms_path.exists():
                self.logger.error(f"NeRF transforms file not found at {transforms_path}")
                return
            
            with open(transforms_path, 'r') as f:
                data = json.load(f)
            
            for frame in data['frames']:
                file_path = os.path.basename(frame['file_path'])
                transform_matrix = np.array(frame['transform_matrix'])
                
                # 提取相機參數
                self.nerf_cameras[file_path] = {
                    'file_path': file_path,
                    'transform_matrix': transform_matrix,
                    'camera_angle_x': data.get('camera_angle_x', 0),
                    'camera_angle_y': data.get('camera_angle_y', 0)
                }
            
            self.logger.info(f"Loaded {len(self.nerf_cameras)} NeRF cameras")
            
        except Exception as e:
            self.logger.error(f"Error loading NeRF data: {str(e)}")
    
    def _load_raw_images(self):
        """加載原始圖像"""
        try:
            for image_path in self.raw_images_path.glob("*.png"):
                try:
                    with Image.open(image_path) as img:
                        self.raw_images[image_path.name] = {
                            'size': img.size,
                            'mode': img.mode
                        }
                except Exception as e:
                    self.logger.warning(f"Error loading image {image_path}: {str(e)}")
            
            self.logger.info(f"Loaded {len(self.raw_images)} raw images")
            
        except Exception as e:
            self.logger.error(f"Error loading raw images: {str(e)}")
    
    def validate_data(self):
        """執行數據驗證"""
        validation_results = {
            'colmap_data': self._validate_colmap_data(),
            'nerf_data': self._validate_nerf_data(),
            'raw_images': self._validate_raw_images(),
            'consistency': self._validate_consistency()
        }
        return validation_results
    
    def _validate_colmap_data(self):
        """驗證 COLMAP 數據"""
        results = {
            'status': 'success',
            'issues': []
        }
        
        if not self.colmap_cameras:
            results['status'] = 'error'
            results['issues'].append('No COLMAP cameras found')
        
        if not self.colmap_images:
            results['status'] = 'error'
            results['issues'].append('No COLMAP images found')
        
        return results
    
    def _validate_nerf_data(self):
        """驗證 NeRF 數據"""
        results = {
            'status': 'success',
            'issues': []
        }
        
        if not self.nerf_cameras:
            results['status'] = 'error'
            results['issues'].append('No NeRF cameras found')
        
        return results
    
    def _validate_raw_images(self):
        """驗證原始圖像"""
        results = {
            'status': 'success',
            'issues': []
        }
        
        if not self.raw_images:
            results['status'] = 'error'
            results['issues'].append('No raw images found')
        
        return results
    
    def _validate_consistency(self):
        """驗證數據一致性"""
        results = {
            'status': 'success',
            'issues': []
        }
        
        # 檢查圖像數量一致性
        colmap_image_count = len(self.colmap_images)
        nerf_camera_count = len(self.nerf_cameras)
        raw_image_count = len(self.raw_images)
        
        if colmap_image_count != nerf_camera_count:
            results['status'] = 'warning'
            results['issues'].append(f'Image count mismatch: COLMAP ({colmap_image_count}) vs NeRF ({nerf_camera_count})')
        
        if colmap_image_count != raw_image_count:
            results['status'] = 'warning'
            results['issues'].append(f'Image count mismatch: COLMAP ({colmap_image_count}) vs Raw ({raw_image_count})')
        
        return results

    def validate_all(self):
        """執行所有驗證檢查"""
        results = {
            'camera_consistency': self.validate_camera_consistency(),
            'image_quality': self.validate_image_quality(),
            'data_completeness': self.validate_data_completeness()
        }
        return results
        
    def validate_camera_consistency(self):
        """驗證相機一致性"""
        results = []
        
        # 檢查相機數量（比對 NeRF 與 COLMAP images 數量）
        nerf_cameras = len(self.nerf_cameras)
        colmap_images = len(self.colmap_images)
        
        if nerf_cameras != colmap_images:
            results.append({
                'title': '相機數量不匹配',
                'message': f'NeRF 相機數量 ({nerf_cameras}) 與 COLMAP 姿態數量 ({colmap_images}) 不一致',
                'status': 'error',
                'details': {
                    'nerf_cameras': nerf_cameras,
                    'colmap_images': colmap_images
                }
            })
        else:
            results.append({
                'title': '相機數量一致性檢查',
                'message': f'✅ NeRF 相機數量 ({nerf_cameras}) 與 COLMAP 姿態數量 ({colmap_images}) 一致',
                'status': 'success',
                'details': {
                    'nerf_cameras': nerf_cameras,
                    'colmap_images': colmap_images
                }
            })
            
        return results
        
    def validate_image_quality(self):
        """驗證圖像質量"""
        results = []
        processed_images = 0
        quality_issues = 0
        
        for frame in self.nerf_cameras.values():
            image_path = self.raw_images_path / frame['file_path']
            if not image_path.exists():
                quality_issues += 1
                results.append({
                    'title': '圖像文件缺失',
                    'message': f'找不到圖像文件: {frame["file_path"]}',
                    'status': 'error',
                    'details': {
                        'file_path': frame['file_path']
                    }
                })
                continue
                
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    raise ValueError("無法讀取圖像")
                
                processed_images += 1
                    
                # 檢查圖像亮度
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                
                if mean_brightness < 30:
                    quality_issues += 1
                    results.append({
                        'title': '圖像亮度過低',
                        'message': f'圖像 {frame["file_path"]} 的平均亮度過低: {mean_brightness:.1f}',
                        'status': 'warning',
                        'details': {
                            'file_path': frame['file_path'],
                            'mean_brightness': float(mean_brightness)
                        }
                    })
                    
                # 檢查圖像對比度
                contrast = np.std(gray)
                if contrast < 30:
                    quality_issues += 1
                    results.append({
                        'title': '圖像對比度過低',
                        'message': f'圖像 {frame["file_path"]} 的對比度過低: {contrast:.1f}',
                        'status': 'warning',
                        'details': {
                            'file_path': frame['file_path'],
                            'contrast': float(contrast)
                        }
                    })
                    
            except Exception as e:
                quality_issues += 1
                results.append({
                    'title': '圖像處理錯誤',
                    'message': f'處理圖像 {frame["file_path"]} 時發生錯誤: {str(e)}',
                    'status': 'error',
                    'details': {
                        'file_path': frame['file_path'],
                        'error': str(e)
                    }
                })
        
        # 如果沒有質量問題，添加成功消息
        if quality_issues == 0 and processed_images > 0:
            results.append({
                'title': '圖像質量檢查',
                'message': f'✅ 已檢查 {processed_images} 張圖像，未發現質量問題',
                'status': 'success',
                'details': {
                    'processed_images': processed_images,
                    'quality_issues': quality_issues
                }
            })
                
        return results
        
    def validate_data_completeness(self):
        """驗證數據完整性"""
        results = []
        missing_files = []
        
        # 檢查必要的目錄和文件
        required_paths = [
            (self.colmap_output_path, "COLMAP 輸出目錄"),
            (self.nerf_data_path, "NeRF 數據目錄"),
            (self.raw_images_path, "原始圖像目錄"),
            (self.nerf_data_path / "transforms.json", "NeRF transforms.json 文件"),
            (self.colmap_output_path / "database.db", "COLMAP 數據庫文件")
        ]
        
        for path, desc in required_paths:
            if not path.exists():
                missing_files.append(desc)
                results.append({
                    'title': '缺少必要文件',
                    'message': f'找不到{desc}: {path}',
                    'status': 'error',
                    'details': {
                        'path': str(path),
                        'description': desc
                    }
                })
        
        # 如果沒有缺失文件，添加成功消息
        if not missing_files:
            results.append({
                'title': '數據完整性檢查',
                'message': f'✅ 所有必要的數據文件和目錄都存在',
                'status': 'success',
                'details': {
                    'checked_paths': len(required_paths),
                    'all_present': True
                }
            })
                
        return results 
#!/usr/bin/env python3
"""
COLMAP到NeRF流水線配置文件
包含所有可調整的參數和設置
"""

import os
from pathlib import Path

class ColmapNerfConfig:
    """COLMAP到NeRF流水線配置類"""
    
    # 支持的圖片格式
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # COLMAP參數配置
    COLMAP_CONFIG = {
        # 特徵提取參數
        'feature_extractor': {
            'ImageReader.single_camera': '1',
            'ImageReader.camera_model': 'PINHOLE',
            'SiftExtraction.use_gpu': '1',
            'SiftExtraction.max_image_size': '3200',
            'SiftExtraction.max_num_features': '8192',
            'SiftExtraction.first_octave': '-1',
            'SiftExtraction.num_octaves': '4',
            'SiftExtraction.octave_resolution': '3',
            'SiftExtraction.peak_threshold': '0.02',
            'SiftExtraction.edge_threshold': '10.0'
        },
        
        # 特徵匹配參數
        'matcher': {
            'SiftMatching.use_gpu': '1',
            'SiftMatching.max_ratio': '0.8',
            'SiftMatching.max_distance': '0.7',
            'SiftMatching.cross_check': '1',
            'SiftMatching.max_num_matches': '32768'
        },
        
        # 稀疏重建參數
        'mapper': {
            'Mapper.ba_refine_focal_length': '1',
            'Mapper.ba_refine_principal_point': '0',
            'Mapper.ba_refine_extra_params': '1',
            'Mapper.min_num_matches': '15',
            'Mapper.init_min_num_inliers': '100',
            'Mapper.abs_pose_min_num_inliers': '30',
            'Mapper.abs_pose_min_inlier_ratio': '0.25',
            'Mapper.filter_max_reproj_error': '4.0',
            'Mapper.filter_min_tri_angle': '1.5'
        }
    }
    
    # 環境變量配置 - 解決無頭模式問題
    COLMAP_ENV = {
        'QT_QPA_PLATFORM': 'offscreen',  # 使用離屏渲染
        'DISPLAY': ':99',                # 虛擬顯示
        'QT_LOGGING_RULES': 'qt.qpa.xcb.debug=false'  # 禁用Qt調試信息
    }
    
    # NeRF轉換參數
    NERF_CONFIG = {
        'coordinate_system': {
            'colmap_to_nerf_transform': True,  # 是否進行座標系轉換
            'normalize_poses': True,           # 是否標準化姿態
            'center_poses': True,              # 是否將姿態中心化
            'scale_poses': True                # 是否縮放姿態
        },
        
        'pose_normalization': {
            'scale_percentile': 90,            # 用於計算尺度的百分位數
            'min_scale': 1e-8                  # 最小尺度值，避免除零
        }
    }
    
    # 數據驗證參數
    VALIDATION_CONFIG = {
        'camera_orientation': {
            'max_similarity_threshold': 0.95,  # 相機朝向最大相似度閾值
            'min_alignment_threshold': 0.3     # 相機朝向中心對齊度最小閾值
        },
        
        'pose_quality': {
            'min_position_std': 0.01,          # 位置分佈最小標準差
            'max_position_std': 10.0,          # 位置分佈最大標準差
            'min_frame_count': 10,             # 最小幀數
            'max_frame_count': 1000            # 最大幀數
        }
    }
    
    # 目錄結構配置
    DIRECTORY_STRUCTURE = {
        'images': 'images',
        'colmap_output': 'colmap_output',
        'nerf_data': 'nerf_data',
        'sparse': 'sparse',
        'dense': 'dense',
        'validation': 'validation'
    }
    
    # 文件名配置
    FILE_NAMES = {
        'database': 'database.db',
        'transforms': 'transforms.json',
        'cameras_bin': 'cameras.bin',
        'images_bin': 'images.bin',
        'points3d_bin': 'points3D.bin'
    }
    
    @classmethod
    def get_colmap_env(cls):
        """獲取COLMAP運行環境變量"""
        env = os.environ.copy()
        env.update(cls.COLMAP_ENV)
        return env
    
    @classmethod
    def get_colmap_feature_cmd(cls, database_path, image_path, **kwargs):
        """生成COLMAP特徵提取命令"""
        cmd = ["colmap", "feature_extractor"]
        cmd.extend(["--database_path", str(database_path)])
        cmd.extend(["--image_path", str(image_path)])
        
        # 添加配置參數
        config = cls.COLMAP_CONFIG['feature_extractor'].copy()
        config.update(kwargs)
        
        for key, value in config.items():
            cmd.extend([f"--{key}", str(value)])
        
        return cmd
    
    @classmethod
    def get_colmap_matcher_cmd(cls, database_path, **kwargs):
        """生成COLMAP特徵匹配命令"""
        cmd = ["colmap", "exhaustive_matcher"]
        cmd.extend(["--database_path", str(database_path)])
        
        # 添加配置參數
        config = cls.COLMAP_CONFIG['matcher'].copy()
        config.update(kwargs)
        
        for key, value in config.items():
            cmd.extend([f"--{key}", str(value)])
        
        return cmd
    
    @classmethod
    def get_colmap_mapper_cmd(cls, database_path, image_path, output_path, **kwargs):
        """生成COLMAP稀疏重建命令"""
        cmd = ["colmap", "mapper"]
        cmd.extend(["--database_path", str(database_path)])
        cmd.extend(["--image_path", str(image_path)])
        cmd.extend(["--output_path", str(output_path)])
        
        # 添加配置參數
        config = cls.COLMAP_CONFIG['mapper'].copy()
        config.update(kwargs)
        
        for key, value in config.items():
            cmd.extend([f"--{key}", str(value)])
        
        return cmd
    
    @classmethod
    def create_project_structure(cls, project_dir):
        """創建項目目錄結構"""
        project_path = Path(project_dir)
        
        directories = [
            project_path / cls.DIRECTORY_STRUCTURE['images'],
            project_path / cls.DIRECTORY_STRUCTURE['colmap_output'],
            project_path / cls.DIRECTORY_STRUCTURE['nerf_data'],
            project_path / cls.DIRECTORY_STRUCTURE['colmap_output'] / cls.DIRECTORY_STRUCTURE['sparse'],
            project_path / cls.DIRECTORY_STRUCTURE['colmap_output'] / cls.DIRECTORY_STRUCTURE['dense'],
            project_path / cls.DIRECTORY_STRUCTURE['validation']
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        return {
            'project_dir': project_path,
            'images_dir': project_path / cls.DIRECTORY_STRUCTURE['images'],
            'colmap_output_dir': project_path / cls.DIRECTORY_STRUCTURE['colmap_output'],
            'nerf_data_dir': project_path / cls.DIRECTORY_STRUCTURE['nerf_data'],
            'sparse_dir': project_path / cls.DIRECTORY_STRUCTURE['colmap_output'] / cls.DIRECTORY_STRUCTURE['sparse'],
            'dense_dir': project_path / cls.DIRECTORY_STRUCTURE['colmap_output'] / cls.DIRECTORY_STRUCTURE['dense'],
            'validation_dir': project_path / cls.DIRECTORY_STRUCTURE['validation']
        }
    
    @classmethod
    def get_file_paths(cls, project_dir):
        """獲取重要文件路徑"""
        project_path = Path(project_dir)
        
        return {
            'database': project_path / cls.DIRECTORY_STRUCTURE['colmap_output'] / cls.FILE_NAMES['database'],
            'transforms': project_path / cls.DIRECTORY_STRUCTURE['nerf_data'] / cls.FILE_NAMES['transforms'],
            'cameras_bin': project_path / cls.DIRECTORY_STRUCTURE['colmap_output'] / cls.DIRECTORY_STRUCTURE['sparse'] / '0' / cls.FILE_NAMES['cameras_bin'],
            'images_bin': project_path / cls.DIRECTORY_STRUCTURE['colmap_output'] / cls.DIRECTORY_STRUCTURE['sparse'] / '0' / cls.FILE_NAMES['images_bin'],
            'points3d_bin': project_path / cls.DIRECTORY_STRUCTURE['colmap_output'] / cls.DIRECTORY_STRUCTURE['sparse'] / '0' / cls.FILE_NAMES['points3d_bin']
        }

# 預設配置實例
DEFAULT_CONFIG = ColmapNerfConfig()

# 環境變量配置
def load_config_from_env():
    """從環境變量加載配置"""
    config = ColmapNerfConfig()
    
    # GPU設置
    if os.getenv('COLMAP_USE_GPU', '1') == '0':
        config.COLMAP_CONFIG['feature_extractor']['SiftExtraction.use_gpu'] = '0'
        config.COLMAP_CONFIG['matcher']['SiftMatching.use_gpu'] = '0'
    
    # 圖片尺寸設置
    max_image_size = os.getenv('COLMAP_MAX_IMAGE_SIZE')
    if max_image_size:
        config.COLMAP_CONFIG['feature_extractor']['SiftExtraction.max_image_size'] = max_image_size
    
    # 特徵點數量設置
    max_features = os.getenv('COLMAP_MAX_FEATURES')
    if max_features:
        config.COLMAP_CONFIG['feature_extractor']['SiftExtraction.max_num_features'] = max_features
    
    # 無頭模式設置
    if os.getenv('COLMAP_HEADLESS', '1') == '1':
        config.COLMAP_ENV['QT_QPA_PLATFORM'] = 'offscreen'
    
    return config 
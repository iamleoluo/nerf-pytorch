#!/usr/bin/env python3
"""
COLMAP輸出轉換為NeRF格式
專為露營車數據集優化
"""

import numpy as np
import json
import os
import argparse
from pathlib import Path
import cv2
from scipy.spatial.transform import Rotation
import read_write_model as rwm

def read_cameras_binary(path):
    """讀取COLMAP的cameras.bin文件"""
    return rwm.read_cameras_binary(path)

def read_images_binary(path):
    """讀取COLMAP的images.bin文件"""
    return rwm.read_images_binary(path)

def qvec2rotmat(qvec):
    """四元數轉旋轉矩陣"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def normalize_poses(poses):
    """標準化相機姿態"""
    # 計算場景中心
    centers = poses[:, :3, 3]
    center = np.mean(centers, axis=0)
    
    # 計算場景尺度
    distances = np.linalg.norm(centers - center, axis=1)
    scale = np.percentile(distances, 90)  # 使用90%分位數避免異常值
    
    # 標準化姿態
    for i in range(poses.shape[0]):
        poses[i, :3, 3] = (poses[i, :3, 3] - center) / scale
    
    return poses, center, scale

def colmap_to_nerf_transform(colmap_transform):
    """
    COLMAP座標系轉換為NeRF座標系
    COLMAP: Y向下，Z向前 (computer vision標準)
    NeRF: Y向上，Z向後 (OpenGL標準)
    """
    # 轉換矩陣
    transform_matrix = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    
    return transform_matrix @ colmap_transform @ transform_matrix

def compute_bbox(poses, scale_factor=1.2):
    """計算場景邊界框"""
    centers = poses[:, :3, 3]
    
    # 計算邊界
    min_bounds = np.min(centers, axis=0)
    max_bounds = np.max(centers, axis=0)
    
    # 添加邊距
    center = (min_bounds + max_bounds) / 2
    size = (max_bounds - min_bounds) * scale_factor
    
    bbox_min = center - size / 2
    bbox_max = center + size / 2
    
    return bbox_min.tolist(), bbox_max.tolist()

def colmap_to_nerf(colmap_dir, images_dir, output_file):
    """
    主轉換函數 - 生成標準NeRF格式
    """
    print("🔄 開始轉換COLMAP到NeRF格式...")
    
    # 讀取COLMAP輸出
    cameras_file = os.path.join(colmap_dir, "cameras.bin")
    images_file = os.path.join(colmap_dir, "images.bin")
    
    if not os.path.exists(cameras_file) or not os.path.exists(images_file):
        print("❌ 找不到COLMAP輸出文件")
        return False
    
    cameras = read_cameras_binary(cameras_file)
    images = read_images_binary(images_file)
    
    print(f"📷 找到 {len(cameras)} 個相機，{len(images)} 張圖片")
    
    # 計算相機內參
    camera_id = list(cameras.keys())[0]  # 假設所有圖片使用同一相機
    camera = cameras[camera_id]
    
    if camera.model != 'PINHOLE':
        print(f"❌ 不支持的相機模型: {camera.model}")
        return False
    
    fx, fy, cx, cy = camera.params
    
    # 計算視野角度
    w, h = camera.width, camera.height
    fov_x = 2 * np.arctan(w / (2 * fx))
    
    # 準備NeRF數據結構 (標準格式)
    transforms = {
        "camera_angle_x": fov_x,
        "frames": []
    }
    
    # 收集所有變換矩陣
    poses = []
    valid_images = []
    
    for img_id, img_data in images.items():
        # 檢查圖片文件是否存在
        img_path = os.path.join(images_dir, img_data.name)
        if not os.path.exists(img_path):
            print(f"⚠️ 圖片不存在: {img_data.name}")
            continue
        
        # 構建變換矩陣
        qvec = img_data.qvec
        tvec = img_data.tvec
        
        # COLMAP的旋轉矩陣 (world-to-camera)
        R = qvec2rotmat(qvec).T  # 轉置得到camera-to-world
        t = -R @ tvec  # 轉換平移向量
        
        # 構建4x4變換矩陣
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        # 轉換座標系
        transform = colmap_to_nerf_transform(transform)
        
        poses.append(transform)
        valid_images.append((img_id, img_data))
    
    if len(poses) == 0:
        print("❌ 沒有有效的圖片")
        return False
    
    poses = np.array(poses)
    
    # 標準化姿態
    poses, scene_center, scene_scale = normalize_poses(poses)
    
    print(f"📐 場景中心: {scene_center}")
    print(f"📏 場景尺度: {scene_scale}")
    
    # 構建NeRF格式的幀
    for i, (img_id, img_data) in enumerate(valid_images):
        # 計算旋轉角度 (可選，用於某些NeRF實現)
        rotation_angle = 0.0  # 默認值，可以根據需要計算實際旋轉
        
        frame = {
            "file_path": img_data.name,
            "rotation": rotation_angle,
            "transform_matrix": poses[i].tolist()
        }
        transforms["frames"].append(frame)
    
    # 保存轉換結果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(transforms, f, indent=4)  # 使用4空格縮進匹配標準格式
    
    print(f"✅ 轉換完成，保存到: {output_file}")
    print(f"📊 總共轉換了 {len(transforms['frames'])} 個幀")
    print(f"📐 相機視野角度: {fov_x:.6f} 弧度")
    return True

def main():
    parser = argparse.ArgumentParser(description="將COLMAP輸出轉換為NeRF格式")
    parser.add_argument("--colmap_dir", required=True, help="COLMAP輸出目錄")
    parser.add_argument("--images_dir", required=True, help="圖片目錄")
    parser.add_argument("--output", required=True, help="輸出文件路徑")
    
    args = parser.parse_args()
    
    success = colmap_to_nerf(
        args.colmap_dir,
        args.images_dir,
        args.output
    )
    
    if not success:
        print("❌ 轉換失敗，請檢查輸入文件")
        exit(1)

if __name__ == "__main__":
    main()

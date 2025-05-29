#!/usr/bin/env python3
"""
修正版COLMAP輸出轉換為NeRF格式
修復座標系轉換問題
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
    """四元數轉旋轉矩陣 - COLMAP格式"""
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

def colmap_to_nerf_transform_corrected(qvec, tvec):
    """
    修正版COLMAP到NeRF的座標轉換
    
    Args:
        qvec: COLMAP四元數 [qw, qx, qy, qz] (world-to-camera)
        tvec: COLMAP平移向量 (world-to-camera)
    
    Returns:
        4x4變換矩陣 (camera-to-world, NeRF格式)
    """
    
    # 1. COLMAP四元數轉旋轉矩陣 (world-to-camera)
    R_w2c = qvec2rotmat(qvec)
    t_w2c = tvec
    
    # 2. 轉換為camera-to-world
    R_c2w = R_w2c.T  # 旋轉矩陣的逆就是轉置
    t_c2w = -R_c2w @ t_w2c  # 平移向量的轉換
    
    # 3. 構建4x4變換矩陣 (COLMAP座標系中的camera-to-world)
    transform_colmap = np.eye(4)
    transform_colmap[:3, :3] = R_c2w
    transform_colmap[:3, 3] = t_c2w
    
    # 4. 座標系轉換矩陣 (COLMAP → NeRF)
    # COLMAP: X右, Y下, Z前
    # NeRF:   X右, Y上, Z後
    coord_transform = np.array([
        [1,  0,  0, 0],  # X軸保持不變
        [0, -1,  0, 0],  # Y軸翻轉 (下→上)
        [0,  0, -1, 0],  # Z軸翻轉 (前→後)
        [0,  0,  0, 1]
    ])
    
    # 5. 正確的座標系轉換
    # 需要將變換矩陣從COLMAP座標系轉換到NeRF座標系
    transform_nerf = coord_transform @ transform_colmap @ coord_transform.T
    
    return transform_nerf

def normalize_poses(poses):
    """標準化相機姿態"""
    # 計算場景中心
    centers = poses[:, :3, 3]
    center = np.mean(centers, axis=0)
    
    # 計算場景尺度
    distances = np.linalg.norm(centers - center, axis=1)
    scale = np.percentile(distances, 90)  # 使用90%分位數避免異常值
    
    # 避免除零
    if scale < 1e-8:
        scale = 1.0
    
    # 標準化姿態
    for i in range(poses.shape[0]):
        poses[i, :3, 3] = (poses[i, :3, 3] - center) / scale
    
    return poses, center, scale

def create_filename_mapping(images_dir, colmap_images):
    """
    創建COLMAP記錄的文件名與實際文件名的映射
    
    Args:
        images_dir: 實際圖片目錄
        colmap_images: COLMAP記錄的圖片信息
    
    Returns:
        dict: {colmap_filename: actual_filename}
    """
    import os
    from pathlib import Path
    
    # 獲取實際存在的圖片文件
    actual_files = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for file_path in Path(images_dir).iterdir():
        if file_path.suffix.lower() in image_extensions:
            actual_files.append(file_path.name)
    
    # 獲取COLMAP記錄的文件名
    colmap_files = [img_data.name for img_data in colmap_images.values()]
    
    print(f"📁 實際文件數量: {len(actual_files)}")
    print(f"📁 COLMAP記錄數量: {len(colmap_files)}")
    
    # 創建映射
    filename_mapping = {}
    
    # 如果數量匹配，嘗試按順序映射
    if len(actual_files) == len(colmap_files):
        # 按文件名排序
        actual_files_sorted = sorted(actual_files)
        colmap_files_sorted = sorted(colmap_files)
        
        for colmap_file, actual_file in zip(colmap_files_sorted, actual_files_sorted):
            filename_mapping[colmap_file] = actual_file
            
        print("✅ 按順序創建文件名映射")
    else:
        # 嘗試直接匹配
        for colmap_file in colmap_files:
            if colmap_file in actual_files:
                filename_mapping[colmap_file] = colmap_file
            else:
                # 如果找不到直接匹配，嘗試模糊匹配
                # 這裡可以添加更複雜的匹配邏輯
                print(f"⚠️ 無法映射文件: {colmap_file}")
    
    print(f"📋 成功映射 {len(filename_mapping)} 個文件")
    return filename_mapping

def find_best_colmap_reconstruction(colmap_base_dir):
    """
    自動找到最佳的COLMAP重建結果
    
    Args:
        colmap_base_dir: COLMAP輸出基礎目錄
    
    Returns:
        最佳重建目錄路徑，如果沒有找到則返回None
    """
    import os
    from pathlib import Path
    
    base_path = Path(colmap_base_dir)
    
    # 可能的重建目錄
    possible_dirs = []
    
    # 檢查 sparse/0, sparse/1, ... 格式
    sparse_dir = base_path / "sparse"
    if sparse_dir.exists():
        for subdir in sparse_dir.iterdir():
            if subdir.is_dir() and subdir.name.isdigit():
                cameras_file = subdir / "cameras.bin"
                images_file = subdir / "images.bin"
                if cameras_file.exists() and images_file.exists():
                    possible_dirs.append(subdir)
    
    # 檢查 0, 1, ... 格式（直接在colmap_output下）
    for subdir in base_path.iterdir():
        if subdir.is_dir() and subdir.name.isdigit():
            cameras_file = subdir / "cameras.bin"
            images_file = subdir / "images.bin"
            if cameras_file.exists() and images_file.exists():
                possible_dirs.append(subdir)
    
    if not possible_dirs:
        print("❌ 沒有找到有效的COLMAP重建結果")
        return None
    
    # 選擇最佳重建（通常是圖片數量最多的）
    best_dir = None
    max_images = 0
    
    for recon_dir in possible_dirs:
        try:
            images_file = recon_dir / "images.bin"
            images = rwm.read_images_binary(str(images_file))
            num_images = len(images)
            
            print(f"📁 重建目錄 {recon_dir.name}: {num_images} 張圖片")
            
            if num_images > max_images:
                max_images = num_images
                best_dir = recon_dir
        except Exception as e:
            print(f"⚠️ 無法讀取重建目錄 {recon_dir.name}: {e}")
            continue
    
    if best_dir:
        print(f"✅ 選擇最佳重建: {best_dir} ({max_images} 張圖片)")
        return str(best_dir)
    else:
        print("❌ 沒有找到有效的重建結果")
        return None

def colmap_to_nerf_fixed(colmap_dir, images_dir, output_file):
    """
    修正版轉換函數 - 生成標準NeRF格式，支持智能文件名映射
    """
    print("🔄 開始修正版COLMAP到NeRF轉換...")
    
    # 自動找到最佳重建結果
    best_recon_dir = find_best_colmap_reconstruction(colmap_dir)
    if not best_recon_dir:
        return False
    
    # 讀取COLMAP輸出
    cameras_file = os.path.join(best_recon_dir, "cameras.bin")
    images_file = os.path.join(best_recon_dir, "images.bin")
    
    if not os.path.exists(cameras_file) or not os.path.exists(images_file):
        print("❌ 找不到COLMAP輸出文件")
        return False
    
    cameras = read_cameras_binary(cameras_file)
    images = read_images_binary(images_file)
    
    print(f"📷 找到 {len(cameras)} 個相機，{len(images)} 張圖片")
    
    # 創建文件名映射
    filename_mapping = create_filename_mapping(images_dir, images)
    
    # 計算相機內參
    camera_id = list(cameras.keys())[0]
    camera = cameras[camera_id]
    
    if camera.model != 'PINHOLE':
        print(f"❌ 不支持的相機模型: {camera.model}")
        return False
    
    fx, fy, cx, cy = camera.params
    w, h = camera.width, camera.height
    fov_x = 2 * np.arctan(w / (2 * fx))
    
    # 準備NeRF數據結構
    transforms = {
        "camera_angle_x": fov_x,
        "frames": []
    }
    
    # 收集所有變換矩陣
    poses = []
    valid_images = []
    
    for img_id, img_data in images.items():
        # 使用映射獲取實際文件名
        colmap_filename = img_data.name
        actual_filename = filename_mapping.get(colmap_filename)
        
        if actual_filename is None:
            print(f"⚠️ 無法找到文件映射: {colmap_filename}")
            continue
            
        # 檢查實際圖片文件是否存在
        img_path = os.path.join(images_dir, actual_filename)
        if not os.path.exists(img_path):
            print(f"⚠️ 圖片不存在: {actual_filename}")
            continue
        
        # 使用修正的轉換函數
        transform = colmap_to_nerf_transform_corrected(img_data.qvec, img_data.tvec)
        
        poses.append(transform)
        valid_images.append((img_id, img_data, actual_filename))
    
    if len(poses) == 0:
        print("❌ 沒有有效的圖片")
        return False
    
    poses = np.array(poses)
    
    # 標準化姿態
    poses, scene_center, scene_scale = normalize_poses(poses)
    
    print(f"📐 場景中心: {scene_center}")
    print(f"📏 場景尺度: {scene_scale}")
    
    # 構建NeRF格式的幀
    for i, (img_id, img_data, actual_filename) in enumerate(valid_images):
        frame = {
            "file_path": actual_filename,  # 使用實際文件名
            "rotation": 0.0,
            "transform_matrix": poses[i].tolist()
        }
        transforms["frames"].append(frame)
    
    # 保存轉換結果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(transforms, f, indent=4)
    
    print(f"✅ 修正版轉換完成，保存到: {output_file}")
    print(f"📊 總共轉換了 {len(transforms['frames'])} 個幀")
    print(f"📐 相機視野角度: {fov_x:.6f} 弧度 ({np.degrees(fov_x):.2f}°)")
    
    # 驗證轉換結果
    verify_conversion_quality(poses)
    
    return True

def verify_conversion_quality(poses):
    """驗證轉換品質"""
    print("\n🔍 轉換品質驗證:")
    
    positions = poses[:, :3, 3]
    directions = -poses[:, :3, 2]  # NeRF中相機朝向-Z
    
    # 檢查相機朝向分佈
    directions_norm = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    dot_products = np.dot(directions_norm, directions_norm.T)
    np.fill_diagonal(dot_products, -1)
    max_similarity = np.max(dot_products)
    
    print(f"  相機朝向最大相似度: {max_similarity:.3f}")
    
    # 檢查位置分佈
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    
    print(f"  位置分佈標準差: {distances.std():.3f}")
    print(f"  平均距離中心: {distances.mean():.3f}")
    
    # 檢查相機朝向是否合理 (應該大部分朝向場景中心)
    to_center = center - positions
    to_center_norm = to_center / np.linalg.norm(to_center, axis=1, keepdims=True)
    
    # 計算相機朝向與朝向中心的相似度
    center_alignment = np.sum(directions_norm * to_center_norm, axis=1)
    avg_alignment = np.mean(center_alignment)
    
    print(f"  相機朝向中心對齊度: {avg_alignment:.3f} (越接近1越好)")
    
    if max_similarity < 0.95:
        print("  ✅ 相機朝向多樣性良好")
    else:
        print("  ⚠️ 相機朝向過於相似")
    
    if avg_alignment > 0.3:
        print("  ✅ 相機大致朝向場景中心")
    else:
        print("  ⚠️ 相機朝向可能有問題")

def main():
    parser = argparse.ArgumentParser(description="修正版COLMAP到NeRF轉換")
    parser.add_argument("--colmap_dir", required=True, help="COLMAP輸出目錄")
    parser.add_argument("--images_dir", required=True, help="圖片目錄")
    parser.add_argument("--output", required=True, help="輸出文件路徑")
    
    args = parser.parse_args()
    
    success = colmap_to_nerf_fixed(
        args.colmap_dir,
        args.images_dir,
        args.output
    )
    
    if not success:
        print("❌ 轉換失敗，請檢查輸入文件")
        exit(1)

if __name__ == "__main__":
    main() 
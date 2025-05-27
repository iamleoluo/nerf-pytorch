#!/usr/bin/env python3
"""
可視化露營車數據集的相機位置
幫助診斷相機姿態是否正確
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_camera_poses(transforms_file):
    """載入相機姿態數據"""
    with open(transforms_file, 'r') as f:
        data = json.load(f)
    
    poses = []
    file_paths = []
    
    for frame in data['frames']:
        transform_matrix = np.array(frame['transform_matrix'])
        poses.append(transform_matrix)
        file_paths.append(frame['file_path'])
    
    return np.array(poses), file_paths, data.get('camera_angle_x', 0)

def extract_camera_positions_and_directions(poses):
    """提取相機位置和朝向"""
    positions = poses[:, :3, 3]  # 位置 (x, y, z)
    
    # 相機朝向 (z軸方向，但要取負值因為相機朝向是-z)
    directions = -poses[:, :3, 2]
    
    # 相機上方向 (y軸方向)
    up_vectors = poses[:, :3, 1]
    
    return positions, directions, up_vectors

def plot_camera_poses(poses, file_paths, title="Camera Poses Visualization"):
    """繪製相機位置和朝向"""
    positions, directions, up_vectors = extract_camera_positions_and_directions(poses)
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D視圖
    ax1 = fig.add_subplot(221, projection='3d')
    
    # 繪製相機位置
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               c=range(len(positions)), cmap='viridis', s=50)
    
    # 繪製相機朝向
    scale = 0.1
    ax1.quiver(positions[:, 0], positions[:, 1], positions[:, 2],
              directions[:, 0], directions[:, 1], directions[:, 2],
              length=scale, color='red', alpha=0.7)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Camera Positions and Orientations')
    
    # XY平面視圖
    ax2 = fig.add_subplot(222)
    ax2.scatter(positions[:, 0], positions[:, 1], c=range(len(positions)), cmap='viridis')
    ax2.quiver(positions[:, 0], positions[:, 1], 
              directions[:, 0], directions[:, 1], 
              scale=5, color='red', alpha=0.7)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Plane View (Top View)')
    ax2.grid(True)
    ax2.axis('equal')
    
    # XZ平面視圖
    ax3 = fig.add_subplot(223)
    ax3.scatter(positions[:, 0], positions[:, 2], c=range(len(positions)), cmap='viridis')
    ax3.quiver(positions[:, 0], positions[:, 2], 
              directions[:, 0], directions[:, 2], 
              scale=5, color='red', alpha=0.7)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Plane View (Side View)')
    ax3.grid(True)
    ax3.axis('equal')
    
    # YZ平面視圖
    ax4 = fig.add_subplot(224)
    ax4.scatter(positions[:, 1], positions[:, 2], c=range(len(positions)), cmap='viridis')
    ax4.quiver(positions[:, 1], positions[:, 2], 
              directions[:, 1], directions[:, 2], 
              scale=5, color='red', alpha=0.7)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('YZ Plane View (Front View)')
    ax4.grid(True)
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.show()
    
    return positions, directions

def analyze_camera_distribution(positions, directions):
    """分析相機分佈"""
    print("📊 相機位置分析")
    print("=" * 50)
    
    print(f"相機數量: {len(positions)}")
    print(f"位置範圍:")
    print(f"  X: {positions[:, 0].min():.3f} ~ {positions[:, 0].max():.3f}")
    print(f"  Y: {positions[:, 1].min():.3f} ~ {positions[:, 1].max():.3f}")
    print(f"  Z: {positions[:, 2].min():.3f} ~ {positions[:, 2].max():.3f}")
    
    # 計算相機間距離
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    print(f"\n距離中心點:")
    print(f"  平均距離: {distances.mean():.3f}")
    print(f"  最小距離: {distances.min():.3f}")
    print(f"  最大距離: {distances.max():.3f}")
    print(f"  標準差: {distances.std():.3f}")
    
    # 檢查是否有重複或過近的相機
    min_distance = float('inf')
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_distance = min(min_distance, dist)
    
    print(f"\n相機間最小距離: {min_distance:.6f}")
    if min_distance < 0.001:
        print("⚠️  警告: 發現相機位置過於接近或重複!")
    
    # 檢查朝向分佈
    directions_norm = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    dot_products = np.dot(directions_norm, directions_norm.T)
    
    # 找出朝向最相似的相機對
    np.fill_diagonal(dot_products, -1)  # 忽略自己
    max_similarity = np.max(dot_products)
    print(f"\n相機朝向最大相似度: {max_similarity:.3f}")
    if max_similarity > 0.99:
        print("⚠️  警告: 發現相機朝向過於相似!")

def main():
    """主函數"""
    print("🎥 露營車數據集相機位置可視化")
    print("=" * 50)
    
    # 載入不同的transforms文件
    datasets = [
        ("data/nerf_synthetic/camper/transforms_train.json", "Training Set Camera Poses"),
        ("data/nerf_synthetic/camper/transforms_val.json", "Validation Set Camera Poses"),
        ("data/nerf_synthetic/camper/transforms_test.json", "Test Set Camera Poses"),
        ("data/nerf_synthetic/camper/transforms.json", "Complete Dataset Camera Poses")
    ]
    
    for transforms_file, title in datasets:
        try:
            print(f"\n📁 載入: {transforms_file}")
            poses, file_paths, camera_angle_x = load_camera_poses(transforms_file)
            print(f"相機角度 (camera_angle_x): {camera_angle_x:.6f} 弧度 ({np.degrees(camera_angle_x):.2f}°)")
            
            positions, directions = plot_camera_poses(poses, file_paths, title)
            analyze_camera_distribution(positions, directions)
            
            # 保存圖片
            plt.savefig(f'camera_poses_{transforms_file.split("/")[-1].replace(".json", "")}.png', 
                       dpi=300, bbox_inches='tight')
            print(f"✅ 圖片已保存: camera_poses_{transforms_file.split('/')[-1].replace('.json', '')}.png")
            
        except FileNotFoundError:
            print(f"❌ 文件不存在: {transforms_file}")
        except Exception as e:
            print(f"❌ 處理錯誤: {e}")

if __name__ == "__main__":
    main() 
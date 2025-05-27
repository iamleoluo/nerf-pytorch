#!/usr/bin/env python3
"""
詳細檢查COLMAP到NeRF的座標系轉換
分析轉換公式是否正確
工具版本 - 放置在tools/analysis目錄
"""

import numpy as np
import json
import os
import sys

# 添加項目根目錄到路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(project_root)

def print_coordinate_systems():
    """詳細說明座標系統差異"""
    print("🔍 座標系統詳細分析")
    print("=" * 60)
    
    print("\n📐 COLMAP座標系統 (Computer Vision標準):")
    print("  X軸: 向右")
    print("  Y軸: 向下")  
    print("  Z軸: 向前 (相機朝向)")
    print("  右手座標系")
    
    print("\n📐 NeRF座標系統 (OpenGL標準):")
    print("  X軸: 向右")
    print("  Y軸: 向上")
    print("  Z軸: 向後 (遠離相機)")
    print("  右手座標系")
    
    print("\n🔄 轉換需求:")
    print("  Y軸: 需要翻轉 (向下 → 向上)")
    print("  Z軸: 需要翻轉 (向前 → 向後)")
    print("  X軸: 保持不變")

def analyze_current_conversion():
    """分析當前轉換公式"""
    print("\n🔧 當前轉換公式分析")
    print("=" * 60)
    
    print("\n當前使用的轉換矩陣:")
    current_transform = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    print(current_transform)
    
    print("\n這個轉換的含義:")
    print("  [1,  0,  0]: X軸保持不變")
    print("  [0, -1,  0]: Y軸翻轉 (向下變向上)")
    print("  [0,  0, -1]: Z軸翻轉 (向前變向後)")
    
    return current_transform

def test_conversion_with_examples():
    """用具體例子測試轉換"""
    print("\n🧪 轉換測試")
    print("=" * 60)
    
    # 測試向量
    test_vectors = {
        "相機朝向前方": np.array([0, 0, 1, 1]),  # COLMAP中相機朝向+Z
        "相機朝向上方": np.array([0, -1, 0, 1]), # COLMAP中相機朝向-Y
        "相機朝向右方": np.array([1, 0, 0, 1]),  # COLMAP中相機朝向+X
        "相機位置": np.array([1, 2, 3, 1])       # 任意位置
    }
    
    current_transform = analyze_current_conversion()
    
    print("\n轉換結果:")
    for name, vector in test_vectors.items():
        transformed = current_transform @ vector
        print(f"  {name}:")
        print(f"    COLMAP: [{vector[0]:6.1f}, {vector[1]:6.1f}, {vector[2]:6.1f}]")
        print(f"    NeRF:   [{transformed[0]:6.1f}, {transformed[1]:6.1f}, {transformed[2]:6.1f}]")

def check_quaternion_to_rotation():
    """檢查四元數到旋轉矩陣的轉換"""
    print("\n🔄 四元數轉換檢查")
    print("=" * 60)
    
    print("COLMAP四元數格式: [qw, qx, qy, qz]")
    print("標準四元數格式: [qx, qy, qz, qw]")
    
    def qvec2rotmat_colmap(qvec):
        """COLMAP的四元數轉旋轉矩陣 (當前使用)"""
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
    
    def qvec2rotmat_standard(qvec):
        """標準四元數轉旋轉矩陣"""
        qw, qx, qy, qz = qvec[0], qvec[1], qvec[2], qvec[3]
        return np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
    
    # 測試四元數 (單位四元數，無旋轉)
    test_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [qw, qx, qy, qz]
    
    R_colmap = qvec2rotmat_colmap(test_quat)
    R_standard = qvec2rotmat_standard(test_quat)
    
    print(f"\n測試四元數: {test_quat}")
    print("COLMAP方法結果:")
    print(R_colmap)
    print("標準方法結果:")
    print(R_standard)
    print(f"是否相同: {np.allclose(R_colmap, R_standard)}")

def analyze_camera_to_world_conversion():
    """分析相機到世界座標的轉換"""
    print("\n🌍 相機到世界座標轉換")
    print("=" * 60)
    
    print("COLMAP存儲格式:")
    print("  qvec: 四元數 (world-to-camera旋轉)")
    print("  tvec: 平移向量 (world-to-camera)")
    
    print("\nNeRF需要格式:")
    print("  transform_matrix: 4x4矩陣 (camera-to-world)")
    
    print("\n轉換步驟:")
    print("1. R_w2c = qvec2rotmat(qvec)  # world-to-camera旋轉")
    print("2. t_w2c = tvec               # world-to-camera平移")
    print("3. R_c2w = R_w2c.T           # camera-to-world旋轉")
    print("4. t_c2w = -R_c2w @ t_w2c    # camera-to-world平移")
    
    print("\n⚠️ 常見錯誤:")
    print("1. 忘記轉置旋轉矩陣")
    print("2. 平移向量轉換錯誤")
    print("3. 座標系轉換順序錯誤")

def propose_corrected_conversion():
    """提出修正的轉換方法"""
    print("\n✅ 建議的正確轉換流程")
    print("=" * 60)
    
    print("def colmap_to_nerf_corrected(qvec, tvec):")
    print("    # 1. COLMAP四元數轉旋轉矩陣 (world-to-camera)")
    print("    R_w2c = qvec2rotmat(qvec)")
    print("    t_w2c = tvec")
    print("    ")
    print("    # 2. 轉換為camera-to-world")
    print("    R_c2w = R_w2c.T")
    print("    t_c2w = -R_c2w @ t_w2c")
    print("    ")
    print("    # 3. 構建4x4變換矩陣")
    print("    transform = np.eye(4)")
    print("    transform[:3, :3] = R_c2w")
    print("    transform[:3, 3] = t_c2w")
    print("    ")
    print("    # 4. 座標系轉換 (COLMAP → NeRF)")
    print("    coord_transform = np.array([")
    print("        [1,  0,  0, 0],")
    print("        [0, -1,  0, 0],")
    print("        [0,  0, -1, 0],")
    print("        [0,  0,  0, 1]")
    print("    ])")
    print("    ")
    print("    # 5. 應用座標系轉換")
    print("    final_transform = coord_transform @ transform @ coord_transform")
    print("    return final_transform")

def check_dataset_transforms(dataset_path=None):
    """檢查數據集中的變換矩陣"""
    print("\n📊 數據集變換矩陣檢查")
    print("=" * 60)
    
    # 使用絕對路徑
    if dataset_path is None:
        dataset_path = os.path.join(project_root, "data/nerf_synthetic/camper_fixed/transforms.json")
    
    if not os.path.exists(dataset_path):
        print(f"❌ 找不到transforms.json文件: {dataset_path}")
        return
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # 檢查前幾個相機的變換矩陣
    print(f"檢查前3個相機的變換矩陣:")
    for i, frame in enumerate(data['frames'][:3]):
        transform = np.array(frame['transform_matrix'])
        
        print(f"\n相機 {i+1} ({frame['file_path']}):")
        print("變換矩陣:")
        print(transform)
        
        # 提取位置和朝向
        position = transform[:3, 3]
        forward = -transform[:3, 2]  # NeRF中相機朝向-Z
        up = transform[:3, 1]
        right = transform[:3, 0]
        
        print(f"位置: [{position[0]:6.3f}, {position[1]:6.3f}, {position[2]:6.3f}]")
        print(f"朝向: [{forward[0]:6.3f}, {forward[1]:6.3f}, {forward[2]:6.3f}]")
        print(f"上方: [{up[0]:6.3f}, {up[1]:6.3f}, {up[2]:6.3f}]")
        
        # 檢查是否為正交矩陣
        R = transform[:3, :3]
        is_orthogonal = np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        det = np.linalg.det(R)
        print(f"旋轉矩陣正交性: {is_orthogonal}")
        print(f"行列式: {det:.6f} (應該接近1.0)")

def main():
    """主函數"""
    print("🔍 COLMAP到NeRF座標轉換詳細分析")
    print("=" * 80)
    
    print_coordinate_systems()
    analyze_current_conversion()
    test_conversion_with_examples()
    check_quaternion_to_rotation()
    analyze_camera_to_world_conversion()
    propose_corrected_conversion()
    check_dataset_transforms()
    
    print("\n🎯 總結")
    print("=" * 60)
    print("1. 檢查四元數轉換是否正確")
    print("2. 確認world-to-camera到camera-to-world的轉換")
    print("3. 驗證座標系轉換矩陣")
    print("4. 檢查變換矩陣的正交性")
    print("5. 分析相機朝向是否合理")

if __name__ == "__main__":
    main() 
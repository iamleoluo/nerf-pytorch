#!/usr/bin/env python3
"""
分析COLMAP重建品質並提供改進建議
"""

import json
import numpy as np
import os

def analyze_camera_poses(transforms_file):
    """分析相機姿態品質"""
    with open(transforms_file, 'r') as f:
        data = json.load(f)
    
    poses = []
    for frame in data['frames']:
        transform_matrix = np.array(frame['transform_matrix'])
        poses.append(transform_matrix)
    
    poses = np.array(poses)
    positions = poses[:, :3, 3]
    directions = -poses[:, :3, 2]  # 相機朝向
    
    return positions, directions, data

def calculate_pose_quality_metrics(positions, directions):
    """計算姿態品質指標"""
    metrics = {}
    
    # 1. 位置分佈分析
    center = np.mean(positions, axis=0)
    distances_from_center = np.linalg.norm(positions - center, axis=1)
    
    metrics['position_spread'] = {
        'mean_distance': distances_from_center.mean(),
        'std_distance': distances_from_center.std(),
        'min_distance': distances_from_center.min(),
        'max_distance': distances_from_center.max()
    }
    
    # 2. 相機間距離分析
    min_distance = float('inf')
    distances = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances.append(dist)
            min_distance = min(min_distance, dist)
    
    metrics['inter_camera_distances'] = {
        'min_distance': min_distance,
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances)
    }
    
    # 3. 視角多樣性分析
    directions_norm = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    dot_products = np.dot(directions_norm, directions_norm.T)
    
    # 移除對角線元素
    mask = ~np.eye(dot_products.shape[0], dtype=bool)
    similarities = dot_products[mask]
    
    metrics['viewing_diversity'] = {
        'max_similarity': similarities.max(),
        'mean_similarity': similarities.mean(),
        'min_similarity': similarities.min(),
        'std_similarity': similarities.std()
    }
    
    # 4. 基線長度分析 (重要用於立體視覺)
    baseline_lengths = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            baseline = np.linalg.norm(positions[i] - positions[j])
            baseline_lengths.append(baseline)
    
    metrics['baseline_analysis'] = {
        'mean_baseline': np.mean(baseline_lengths),
        'std_baseline': np.std(baseline_lengths),
        'min_baseline': np.min(baseline_lengths),
        'max_baseline': np.max(baseline_lengths)
    }
    
    return metrics

def diagnose_problems(metrics):
    """診斷問題並提供建議"""
    problems = []
    suggestions = []
    
    # 檢查視角多樣性
    if metrics['viewing_diversity']['max_similarity'] > 0.99:
        problems.append("❌ 相機朝向過於相似 (max similarity > 0.99)")
        suggestions.append("📝 建議: 增加更多不同角度的照片，特別是從不同高度和方位拍攝")
    
    if metrics['viewing_diversity']['mean_similarity'] > 0.8:
        problems.append("❌ 整體視角多樣性不足 (mean similarity > 0.8)")
        suggestions.append("📝 建議: 拍攝時應該圍繞物體360度移動，並改變高度")
    
    # 檢查相機間距離
    if metrics['inter_camera_distances']['min_distance'] < 0.05:
        problems.append("❌ 相機位置過於接近 (min distance < 0.05)")
        suggestions.append("📝 建議: 增加相機間的距離，避免連續照片過於相似")
    
    # 檢查基線長度
    if metrics['baseline_analysis']['mean_baseline'] < 0.1:
        problems.append("❌ 基線長度過短，影響深度估計")
        suggestions.append("📝 建議: 增加相機移動距離，提供更好的立體視覺線索")
    
    # 檢查位置分佈
    if metrics['position_spread']['std_distance'] < 0.1:
        problems.append("❌ 相機位置分佈過於集中")
        suggestions.append("📝 建議: 從更多不同距離拍攝物體")
    
    return problems, suggestions

def suggest_improvements():
    """提供具體的改進建議"""
    print("\n🎯 COLMAP重建品質改進建議")
    print("=" * 60)
    
    print("\n📸 拍攝技巧改進:")
    print("1. 圍繞露營車360度拍攝，每15-30度拍一張")
    print("2. 從不同高度拍攝 (蹲下、正常高度、舉高)")
    print("3. 改變與露營車的距離 (近景、中景、遠景)")
    print("4. 確保相鄰照片有60-80%的重疊")
    print("5. 避免純平移，增加一些旋轉角度")
    
    print("\n🔧 COLMAP參數調整:")
    print("1. 增加特徵點數量: --SiftExtraction.max_num_features 8192")
    print("2. 調整匹配參數: --SiftMatching.max_ratio 0.8")
    print("3. 使用更嚴格的Bundle Adjustment")
    print("4. 考慮使用mask來排除天空等無紋理區域")
    
    print("\n⚙️ NeRF訓練參數調整:")
    print("1. 降低factor值 (從8改為2或4)")
    print("2. 增加訓練步數")
    print("3. 調整學習率策略")
    print("4. 考慮使用更多的採樣點")

def main():
    """主函數"""
    print("🔍 COLMAP重建品質分析")
    print("=" * 50)
    
    transforms_file = "data/nerf_synthetic/camper/transforms.json"
    
    if not os.path.exists(transforms_file):
        print(f"❌ 文件不存在: {transforms_file}")
        return
    
    # 載入和分析數據
    positions, directions, data = analyze_camera_poses(transforms_file)
    metrics = calculate_pose_quality_metrics(positions, directions)
    
    # 顯示基本信息
    print(f"\n📊 基本信息:")
    print(f"相機數量: {len(positions)}")
    print(f"相機角度: {data.get('camera_angle_x', 0):.6f} 弧度 ({np.degrees(data.get('camera_angle_x', 0)):.2f}°)")
    
    # 顯示詳細指標
    print(f"\n📏 位置分佈指標:")
    pos_metrics = metrics['position_spread']
    print(f"  平均距離中心: {pos_metrics['mean_distance']:.3f}")
    print(f"  距離標準差: {pos_metrics['std_distance']:.3f}")
    print(f"  距離範圍: {pos_metrics['min_distance']:.3f} ~ {pos_metrics['max_distance']:.3f}")
    
    print(f"\n📐 相機間距離:")
    dist_metrics = metrics['inter_camera_distances']
    print(f"  最小距離: {dist_metrics['min_distance']:.6f}")
    print(f"  平均距離: {dist_metrics['mean_distance']:.3f}")
    print(f"  距離標準差: {dist_metrics['std_distance']:.3f}")
    
    print(f"\n👁️ 視角多樣性:")
    view_metrics = metrics['viewing_diversity']
    print(f"  最大相似度: {view_metrics['max_similarity']:.6f}")
    print(f"  平均相似度: {view_metrics['mean_similarity']:.3f}")
    print(f"  最小相似度: {view_metrics['min_similarity']:.3f}")
    
    print(f"\n📏 基線分析:")
    baseline_metrics = metrics['baseline_analysis']
    print(f"  平均基線: {baseline_metrics['mean_baseline']:.3f}")
    print(f"  基線標準差: {baseline_metrics['std_baseline']:.3f}")
    print(f"  基線範圍: {baseline_metrics['min_baseline']:.3f} ~ {baseline_metrics['max_baseline']:.3f}")
    
    # 診斷問題
    problems, suggestions = diagnose_problems(metrics)
    
    if problems:
        print(f"\n🚨 發現的問題:")
        for problem in problems:
            print(f"  {problem}")
        
        print(f"\n💡 改進建議:")
        for suggestion in suggestions:
            print(f"  {suggestion}")
    else:
        print(f"\n✅ 相機姿態品質良好!")
    
    # 提供具體改進建議
    suggest_improvements()

if __name__ == "__main__":
    main() 
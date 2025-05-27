#!/usr/bin/env python3
"""
相機位置和朝向視覺化工具
分析NeRF數據集中的相機分佈
工具版本 - 放置在tools/analysis目錄
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import datetime

# 添加項目根目錄到路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(project_root)

def load_transforms(dataset_path):
    """載入transforms.json文件"""
    if not os.path.exists(dataset_path):
        print(f"❌ 找不到數據集文件: {dataset_path}")
        return None
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"✅ 載入數據集，包含 {len(data['frames'])} 個相機")
    return data

def extract_camera_poses(transforms_data):
    """提取相機位置和朝向"""
    positions = []
    directions = []
    up_vectors = []
    
    for frame in transforms_data['frames']:
        transform = np.array(frame['transform_matrix'])
        
        # 提取位置
        position = transform[:3, 3]
        positions.append(position)
        
        # 提取朝向 (NeRF中相機朝向-Z方向)
        direction = -transform[:3, 2]
        directions.append(direction)
        
        # 提取上方向 (Y軸)
        up = transform[:3, 1]
        up_vectors.append(up)
    
    return np.array(positions), np.array(directions), np.array(up_vectors)

def analyze_camera_distribution(positions, directions):
    """分析相機分佈"""
    print("\n📊 相機分佈分析")
    print("=" * 50)
    
    # 位置統計
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    
    print(f"相機位置統計:")
    print(f"  中心點: [{center[0]:6.3f}, {center[1]:6.3f}, {center[2]:6.3f}]")
    print(f"  平均距離中心: {distances.mean():.3f}")
    print(f"  距離標準差: {distances.std():.3f}")
    print(f"  最小距離: {distances.min():.3f}")
    print(f"  最大距離: {distances.max():.3f}")
    
    # 朝向統計
    directions_norm = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    # 計算相機朝向的相似度
    dot_products = np.dot(directions_norm, directions_norm.T)
    np.fill_diagonal(dot_products, -1)  # 排除自己與自己的比較
    max_similarity = np.max(dot_products)
    min_similarity = np.min(dot_products)
    avg_similarity = np.mean(dot_products[dot_products > -1])
    
    print(f"\n相機朝向統計:")
    print(f"  最大相似度: {max_similarity:.3f}")
    print(f"  最小相似度: {min_similarity:.3f}")
    print(f"  平均相似度: {avg_similarity:.3f}")
    
    # 檢查相機是否朝向中心
    to_center = center - positions
    to_center_norm = to_center / np.linalg.norm(to_center, axis=1, keepdims=True)
    center_alignment = np.sum(directions_norm * to_center_norm, axis=1)
    
    print(f"\n相機朝向中心分析:")
    print(f"  平均朝向中心度: {center_alignment.mean():.3f}")
    print(f"  朝向中心度標準差: {center_alignment.std():.3f}")
    print(f"  朝向中心的相機數: {np.sum(center_alignment > 0.5)}/{len(center_alignment)}")
    
    return {
        'center': center,
        'distances': distances,
        'max_similarity': max_similarity,
        'center_alignment': center_alignment.mean()
    }

def visualize_cameras_3d(positions, directions, save_path=None):
    """3D視覺化相機位置和朝向"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 繪製相機位置
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
              c='red', s=50, alpha=0.7, label='相機位置')
    
    # 繪製相機朝向
    scale = 0.1  # 箭頭長度縮放
    ax.quiver(positions[:, 0], positions[:, 1], positions[:, 2],
              directions[:, 0] * scale, directions[:, 1] * scale, directions[:, 2] * scale,
              color='blue', alpha=0.6, label='相機朝向')
    
    # 計算並繪製場景中心
    center = np.mean(positions, axis=0)
    ax.scatter(center[0], center[1], center[2], 
              c='green', s=100, marker='*', label='場景中心')
    
    # 設置軸標籤和標題
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('相機位置和朝向分佈 (NeRF座標系)')
    ax.legend()
    
    # 設置相等的軸比例
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                         positions[:, 1].max() - positions[:, 1].min(),
                         positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 3D視覺化保存到: {save_path}")
    
    plt.show()

def visualize_camera_distribution(positions, directions, save_path=None):
    """2D投影視覺化相機分佈"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # XY平面投影
    axes[0, 0].scatter(positions[:, 0], positions[:, 1], c='red', alpha=0.7)
    axes[0, 0].quiver(positions[:, 0], positions[:, 1], 
                     directions[:, 0], directions[:, 1], 
                     scale=5, alpha=0.6, color='blue')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('XY平面投影')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # XZ平面投影
    axes[0, 1].scatter(positions[:, 0], positions[:, 2], c='red', alpha=0.7)
    axes[0, 1].quiver(positions[:, 0], positions[:, 2], 
                     directions[:, 0], directions[:, 2], 
                     scale=5, alpha=0.6, color='blue')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    axes[0, 1].set_title('XZ平面投影')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # YZ平面投影
    axes[1, 0].scatter(positions[:, 1], positions[:, 2], c='red', alpha=0.7)
    axes[1, 0].quiver(positions[:, 1], positions[:, 2], 
                     directions[:, 1], directions[:, 2], 
                     scale=5, alpha=0.6, color='blue')
    axes[1, 0].set_xlabel('Y')
    axes[1, 0].set_ylabel('Z')
    axes[1, 0].set_title('YZ平面投影')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')
    
    # 距離分佈直方圖
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    axes[1, 1].hist(distances, bins=20, alpha=0.7, color='green')
    axes[1, 1].set_xlabel('距離中心')
    axes[1, 1].set_ylabel('相機數量')
    axes[1, 1].set_title('相機距離中心分佈')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 2D分佈圖保存到: {save_path}")
    
    plt.show()

def generate_quality_report(stats, dataset_name="Unknown"):
    """生成品質報告"""
    print(f"\n📋 數據集品質報告: {dataset_name}")
    print("=" * 60)
    
    # 評估標準
    print("評估標準:")
    print("  相機朝向多樣性: 最大相似度 < 0.95 為良好")
    print("  相機朝向中心: 平均朝向中心度 > 0.3 為良好")
    print("  位置分佈: 距離標準差 > 0.1 為良好")
    
    print("\n評估結果:")
    
    # 朝向多樣性
    if stats['max_similarity'] < 0.95:
        print("  ✅ 相機朝向多樣性: 良好")
    else:
        print("  ⚠️ 相機朝向多樣性: 需要改善")
    
    # 朝向中心
    if stats['center_alignment'] > 0.3:
        print("  ✅ 相機朝向中心: 良好")
    else:
        print("  ⚠️ 相機朝向中心: 需要改善")
    
    # 位置分佈
    if stats['distances'].std() > 0.1:
        print("  ✅ 位置分佈: 良好")
    else:
        print("  ⚠️ 位置分佈: 需要改善")
    
    # 總體評估
    good_count = sum([
        stats['max_similarity'] < 0.95,
        stats['center_alignment'] > 0.3,
        stats['distances'].std() > 0.1
    ])
    
    if good_count == 3:
        print("\n🎉 總體評估: 優秀")
    elif good_count == 2:
        print("\n👍 總體評估: 良好")
    elif good_count == 1:
        print("\n⚠️ 總體評估: 需要改善")
    else:
        print("\n❌ 總體評估: 品質較差")

def evaluate_camera_distribution(positions, directions):
    """評估相機分佈的合理性"""
    print("\n📊 相機分佈合理性評估")
    print("=" * 60)
    
    # 1. 計算相機間距
    n_cameras = len(positions)
    distances = []
    for i in range(n_cameras):
        for j in range(i+1, n_cameras):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances.append(dist)
    
    distances = np.array(distances)
    min_dist = np.min(distances)
    avg_dist = np.mean(distances)
    max_dist = np.max(distances)
    
    print("\n1️⃣ 相機間距分析:")
    print(f"  最小間距: {min_dist:.3f}")
    print(f"  平均間距: {avg_dist:.3f}")
    print(f"  最大間距: {max_dist:.3f}")
    
    # 評估間距合理性
    if min_dist < 0.1:
        print("  ⚠️ 警告: 存在過於接近的相機")
    if max_dist > 10.0:
        print("  ⚠️ 警告: 存在過於分散的相機")
    
    # 2. 分析相機朝向分佈
    directions_norm = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    center = np.mean(positions, axis=0)
    
    # 計算相機朝向與中心點方向的夾角
    to_center = center - positions
    to_center_norm = to_center / np.linalg.norm(to_center, axis=1, keepdims=True)
    center_alignment = np.sum(directions_norm * to_center_norm, axis=1)
    
    print("\n2️⃣ 相機朝向分析:")
    print(f"  平均朝向中心度: {np.mean(center_alignment):.3f}")
    print(f"  朝向中心度標準差: {np.std(center_alignment):.3f}")
    print(f"  朝向中心的相機數: {np.sum(center_alignment > 0.5)}/{len(center_alignment)}")
    
    # 評估朝向合理性
    if np.mean(center_alignment) < 0.3:
        print("  ⚠️ 警告: 相機朝向過於分散")
    if np.sum(center_alignment > 0.5) < len(center_alignment) * 0.5:
        print("  ⚠️ 警告: 朝向中心的相機比例過低")
    
    # 3. 分析覆蓋範圍
    # 計算相機位置的中心和範圍
    pos_center = np.mean(positions, axis=0)
    pos_std = np.std(positions, axis=0)
    
    print("\n3️⃣ 覆蓋範圍分析:")
    print(f"  場景中心: [{pos_center[0]:6.3f}, {pos_center[1]:6.3f}, {pos_center[2]:6.3f}]")
    print(f"  位置標準差: [{pos_std[0]:6.3f}, {pos_std[1]:6.3f}, {pos_std[2]:6.3f}]")
    
    # 評估覆蓋範圍合理性
    if np.any(pos_std < 0.1):
        print("  ⚠️ 警告: 某個維度的覆蓋範圍過小")
    if np.any(pos_std > 5.0):
        print("  ⚠️ 警告: 某個維度的覆蓋範圍過大")
    
    # 4. 綜合評估
    print("\n4️⃣ 綜合評估:")
    
    # 計算各項指標的得分
    spacing_score = 1.0
    if min_dist < 0.1:
        spacing_score -= 0.3
    if max_dist > 10.0:
        spacing_score -= 0.2
    
    orientation_score = 1.0
    if np.mean(center_alignment) < 0.3:
        orientation_score -= 0.3
    if np.sum(center_alignment > 0.5) < len(center_alignment) * 0.5:
        orientation_score -= 0.2
    
    coverage_score = 1.0
    if np.any(pos_std < 0.1):
        coverage_score -= 0.3
    if np.any(pos_std > 5.0):
        coverage_score -= 0.2
    
    total_score = (spacing_score + orientation_score + coverage_score) / 3
    
    print(f"  相機間距得分: {spacing_score:.2f}")
    print(f"  朝向分佈得分: {orientation_score:.2f}")
    print(f"  覆蓋範圍得分: {coverage_score:.2f}")
    print(f"  總體得分: {total_score:.2f}")
    
    if total_score >= 0.8:
        print("  ✅ 相機分佈良好")
    elif total_score >= 0.6:
        print("  ⚠️ 相機分佈一般，建議優化")
    else:
        print("  ❌ 相機分佈較差，需要重新規劃")
    
    return {
        'spacing_score': spacing_score,
        'orientation_score': orientation_score,
        'coverage_score': coverage_score,
        'total_score': total_score
    }

def generate_detailed_report(dataset_name, stats, evaluation, positions, directions, output_dir):
    """生成詳細的分析報告"""
    report_path = os.path.join(output_dir, "camera_analysis_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # 標題
        f.write(f"# 相機分佈分析報告 - {dataset_name}\n\n")
        f.write(f"生成時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 基本統計
        f.write("## 1. 基本統計\n\n")
        f.write(f"- 相機總數: {len(positions)}\n")
        f.write(f"- 場景中心: [{stats['center'][0]:.3f}, {stats['center'][1]:.3f}, {stats['center'][2]:.3f}]\n")
        f.write(f"- 平均距離中心: {stats['distances'].mean():.3f}\n")
        f.write(f"- 距離標準差: {stats['distances'].std():.3f}\n")
        f.write(f"- 最小距離: {stats['distances'].min():.3f}\n")
        f.write(f"- 最大距離: {stats['distances'].max():.3f}\n\n")
        
        # 相機朝向分析
        f.write("## 2. 相機朝向分析\n\n")
        f.write(f"- 最大相似度: {stats['max_similarity']:.3f}\n")
        f.write(f"- 最小相似度: {min(np.dot(directions, directions.T)[np.triu_indices(len(directions), k=1)]):.3f}\n")
        f.write(f"- 平均相似度: {stats['center_alignment']:.3f}\n\n")
        
        # 分佈評估
        f.write("## 3. 分佈評估\n\n")
        f.write("### 3.1 相機間距\n\n")
        f.write(f"- 最小間距: {min(np.linalg.norm(positions[i] - positions[j]) for i in range(len(positions)) for j in range(i+1, len(positions))):.3f}\n")
        f.write(f"- 平均間距: {np.mean([np.linalg.norm(positions[i] - positions[j]) for i in range(len(positions)) for j in range(i+1, len(positions))]):.3f}\n")
        f.write(f"- 最大間距: {max(np.linalg.norm(positions[i] - positions[j]) for i in range(len(positions)) for j in range(i+1, len(positions))):.3f}\n")
        f.write(f"- 間距得分: {evaluation['spacing_score']:.2f}\n\n")
        
        f.write("### 3.2 朝向分佈\n\n")
        directions_norm = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        center = np.mean(positions, axis=0)
        to_center = center - positions
        to_center_norm = to_center / np.linalg.norm(to_center, axis=1, keepdims=True)
        center_alignment = np.sum(directions_norm * to_center_norm, axis=1)
        
        f.write(f"- 平均朝向中心度: {np.mean(center_alignment):.3f}\n")
        f.write(f"- 朝向中心度標準差: {np.std(center_alignment):.3f}\n")
        f.write(f"- 朝向中心的相機數: {np.sum(center_alignment > 0.5)}/{len(center_alignment)}\n")
        f.write(f"- 朝向得分: {evaluation['orientation_score']:.2f}\n\n")
        
        f.write("### 3.3 覆蓋範圍\n\n")
        pos_std = np.std(positions, axis=0)
        f.write(f"- X軸標準差: {pos_std[0]:.3f}\n")
        f.write(f"- Y軸標準差: {pos_std[1]:.3f}\n")
        f.write(f"- Z軸標準差: {pos_std[2]:.3f}\n")
        f.write(f"- 覆蓋得分: {evaluation['coverage_score']:.2f}\n\n")
        
        # 綜合評估
        f.write("## 4. 綜合評估\n\n")
        f.write(f"- 總體得分: {evaluation['total_score']:.2f}\n\n")
        
        if evaluation['total_score'] >= 0.8:
            f.write("### 評估結果: ✅ 相機分佈良好\n\n")
            f.write("相機分佈符合以下要求：\n")
            f.write("- 相機間距適中\n")
            f.write("- 朝向分佈合理\n")
            f.write("- 覆蓋範圍充足\n\n")
        elif evaluation['total_score'] >= 0.6:
            f.write("### 評估結果: ⚠️ 相機分佈一般\n\n")
            f.write("建議優化以下方面：\n")
            if evaluation['spacing_score'] < 0.8:
                f.write("- 調整相機間距，避免過於集中或分散\n")
            if evaluation['orientation_score'] < 0.8:
                f.write("- 優化相機朝向，提高朝向中心的相機比例\n")
            if evaluation['coverage_score'] < 0.8:
                f.write("- 擴大覆蓋範圍，確保各維度都有足夠的覆蓋\n\n")
        else:
            f.write("### 評估結果: ❌ 相機分佈較差\n\n")
            f.write("需要重新規劃相機分佈：\n")
            f.write("- 重新設計相機位置，確保合理的間距\n")
            f.write("- 調整相機朝向，提高朝向中心的比例\n")
            f.write("- 優化覆蓋範圍，確保場景各部分的覆蓋\n\n")
        
        # 視覺化說明
        f.write("## 5. 視覺化說明\n\n")
        f.write("本報告包含以下視覺化圖表：\n")
        f.write("- `cameras_3d.png`: 3D視覺化圖，顯示相機位置和朝向\n")
        f.write("- `cameras_2d.png`: 2D投影圖，顯示相機在不同平面的分佈\n\n")
        
        # 注意事項
        f.write("## 6. 注意事項\n\n")
        f.write("- 本報告基於當前數據集生成，僅供參考\n")
        f.write("- 建議根據實際場景需求調整評估標準\n")
        f.write("- 定期更新相機分佈以確保最佳效果\n")
    
    print(f"📝 詳細報告已保存至: {report_path}")

def main():
    """主函數"""
    print("📷 相機位置和朝向視覺化工具")
    print("=" * 50)
    
    # 使用絕對路徑
    dataset_path = os.path.join(project_root, "data/nerf_synthetic/camper_fixed/transforms.json")
    
    # 載入數據
    transforms_data = load_transforms(dataset_path)
    if transforms_data is None:
        return
    
    # 提取相機姿態
    positions, directions, up_vectors = extract_camera_poses(transforms_data)
    
    # 分析分佈
    stats = analyze_camera_distribution(positions, directions)
    
    # 評估相機分佈合理性
    evaluation = evaluate_camera_distribution(positions, directions)
    
    # 生成品質報告
    generate_quality_report(stats, "camper_fixed")
    
    # 視覺化
    print("\n🎨 生成視覺化圖表...")
    
    # 創建輸出目錄
    output_dir = os.path.join(project_root, "outputs/camera_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # 3D視覺化
    visualize_cameras_3d(positions, directions, 
                        os.path.join(output_dir, "cameras_3d.png"))
    
    # 2D分佈圖
    visualize_camera_distribution(positions, directions,
                                 os.path.join(output_dir, "cameras_2d.png"))
    
    # 生成詳細報告
    generate_detailed_report("camper_fixed", stats, evaluation, positions, directions, output_dir)
    
    print(f"\n✅ 分析完成！圖表和報告保存在: {output_dir}")

if __name__ == "__main__":
    main() 
from flask import Flask, render_template, jsonify, request, send_file
import os
import json
import numpy as np
from pathlib import Path
from validators.data_validator import DataValidator
from PIL import Image
import io
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 配置路徑
PROJECT_ROOT = Path(__file__).parent.parent
COLMAP_OUTPUT = PROJECT_ROOT / "camper_nerf" / "colmap_output"
NERF_DATA = PROJECT_ROOT / "camper_nerf" / "nerf_data"
RAW_IMAGES = PROJECT_ROOT / "camper_nerf" / "images"

# 檢查路徑
for path, name in [
    (COLMAP_OUTPUT, "COLMAP 輸出目錄"),
    (NERF_DATA, "NeRF 數據目錄"),
    (RAW_IMAGES, "原始圖像目錄")
]:
    if not path.exists():
        logger.warning(f"{name}不存在: {path}")
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"已創建{name}: {path}")

# 初始化數據驗證器
try:
    validator = DataValidator(COLMAP_OUTPUT, NERF_DATA, RAW_IMAGES)
    logger.info("數據驗證器初始化成功")
except Exception as e:
    logger.error(f"數據驗證器初始化失敗: {str(e)}")
    raise

@app.route('/')
def index():
    """主頁面"""
    return render_template('index.html')

@app.route('/test')
def test():
    """測試頁面"""
    return render_template('test.html')

@app.route('/debug')
def debug():
    """調試頁面"""
    return render_template('debug.html')

@app.route('/simple')
def simple():
    """簡單測試頁面"""
    return render_template('simple_test.html')

@app.route('/api/cameras')
def get_cameras():
    """獲取相機數據"""
    try:
        transforms_path = NERF_DATA / "transforms.json"
        if not transforms_path.exists():
            return jsonify({
                'status': 'error',
                'message': f'找不到 transforms.json 文件: {transforms_path}'
            }), 404
        with open(transforms_path, 'r') as f:
            nerf_data = json.load(f)
        colmap_images = validator.colmap_images
        cameras = []
        for frame in nerf_data['frames']:
            file_name = os.path.basename(frame['file_path'])
            colmap_cam = colmap_images.get(file_name)
            colmap_transform = None
            colmap_id = None
            
            if colmap_cam:
                # 使用正確的COLMAP數據結構
                qvec = colmap_cam['qvec']  # numpy array [qw, qx, qy, qz]
                tvec = colmap_cam['tvec']  # numpy array [tx, ty, tz]
                
                # 四元數轉旋轉矩陣
                qw, qx, qy, qz = qvec[0], qvec[1], qvec[2], qvec[3]
                tx, ty, tz = tvec[0], tvec[1], tvec[2]
                
                # 歸一化四元數
                q = np.array([qw, qx, qy, qz], dtype=np.float64)
                q = q / np.linalg.norm(q)
                w, x, y, z = q
                
                # 四元數轉旋轉矩陣
                R = np.array([
                    [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                    [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                    [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
                ])
                
                # 構建變換矩陣
                colmap_transform = np.eye(4)
                colmap_transform[:3, :3] = R
                colmap_transform[:3, 3] = [tx, ty, tz]
                colmap_transform = colmap_transform.tolist()
                colmap_id = colmap_cam['image_id']
            camera = {
                'file_path': frame['file_path'],
                'transform_matrix': frame['transform_matrix'],
                'colmap_id': colmap_id,
                'colmap_transform': colmap_transform
            }
            cameras.append(camera)
        return jsonify({
            'status': 'success',
            'data': cameras
        })
    except Exception as e:
        logger.error(f"獲取相機數據失敗: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/image/<path:image_path>')
def get_image(image_path):
    """獲取圖片"""
    try:
        image_path = RAW_IMAGES / image_path
        if not image_path.exists():
            logger.warning(f"找不到圖片文件: {image_path}")
            return jsonify({
                'status': 'error',
                'message': f'找不到圖片文件: {image_path}'
            }), 404
        
        # 讀取圖片
        img = Image.open(image_path)
        
        # 調整圖片大小
        max_size = (800, 800)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # 轉換為字節流
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format)
        img_byte_arr.seek(0)
        
        return send_file(
            img_byte_arr,
            mimetype=f'image/{img.format.lower()}'
        )
    except Exception as e:
        logger.error(f"獲取圖片失敗: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/validate')
def validate_data():
    """驗證數據"""
    try:
        # 執行所有驗證
        results = validator.validate_all()
        
        # 合併所有驗證結果
        all_results = []
        for category, category_results in results.items():
            all_results.extend(category_results)
        
        return jsonify({
            'status': 'success',
            'results': all_results
        })
    except Exception as e:
        logger.error(f"驗證數據失敗: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
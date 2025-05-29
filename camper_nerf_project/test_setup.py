#!/usr/bin/env python3
"""
系統設置測試腳本
檢查COLMAP到NeRF流水線的所有依賴和配置
"""

import sys
import subprocess
import importlib
from pathlib import Path

def test_python_version():
    """測試Python版本"""
    print("🐍 檢查Python版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python版本過低: {version.major}.{version.minor}.{version.micro}")
        print("     需要Python 3.7或更高版本")
        return False

def test_python_packages():
    """測試Python包"""
    print("\n📦 檢查Python依賴包...")
    
    required_packages = {
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'scipy': 'scipy',
        'pathlib': 'pathlib (內建)',
        'argparse': 'argparse (內建)',
        'json': 'json (內建)',
        'subprocess': 'subprocess (內建)',
        'shutil': 'shutil (內建)'
    }
    
    all_good = True
    for module, package in required_packages.items():
        try:
            importlib.import_module(module)
            print(f"  ✅ {module} ({package})")
        except ImportError:
            print(f"  ❌ {module} ({package}) - 未安裝")
            all_good = False
    
    return all_good

def test_colmap():
    """測試COLMAP安裝"""
    print("\n🏗️ 檢查COLMAP...")
    
    try:
        result = subprocess.run(['colmap', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("  ✅ COLMAP已安裝並可用")
            return True
        else:
            print("  ❌ COLMAP執行失敗")
            return False
    except FileNotFoundError:
        print("  ❌ COLMAP未安裝或不在PATH中")
        print("     請參考: https://colmap.github.io/install.html")
        return False
    except subprocess.TimeoutExpired:
        print("  ❌ COLMAP響應超時")
        return False

def test_directory_structure():
    """測試目錄結構"""
    print("\n📁 檢查目錄結構...")
    
    current_dir = Path.cwd()
    required_files = [
        'colmap_nerf_pipeline.py',
        'colmap2nerf_fixed.py',
        'config.py',
        'run_pipeline.sh',
        'requirements.txt',
        'README.md'
    ]
    
    all_good = True
    for file_name in required_files:
        file_path = current_dir / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} - 文件不存在")
            all_good = False
    
    # 檢查目錄
    required_dirs = ['raw_images', 'camper_nerf', 'data_validation']
    for dir_name in required_dirs:
        dir_path = current_dir / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ⚠️ {dir_name}/ - 目錄不存在 (運行時會自動創建)")
    
    return all_good

def test_raw_images():
    """測試原始圖片目錄"""
    print("\n📷 檢查原始圖片...")
    
    raw_images_dir = Path.cwd() / 'raw_images'
    if not raw_images_dir.exists():
        print("  ⚠️ raw_images目錄不存在")
        return False
    
    # 支持的圖片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(raw_images_dir.glob(f'*{ext}'))
        image_files.extend(raw_images_dir.glob(f'*{ext.upper()}'))
    
    if len(image_files) > 0:
        print(f"  ✅ 找到 {len(image_files)} 張圖片")
        return True
    else:
        print("  ⚠️ 沒有找到圖片文件")
        print("     請將圖片放入raw_images目錄")
        return False

def test_config():
    """測試配置文件"""
    print("\n⚙️ 檢查配置...")
    
    try:
        from config import ColmapNerfConfig, load_config_from_env
        config = ColmapNerfConfig()
        print("  ✅ 配置文件加載成功")
        
        # 測試配置方法
        test_cmd = config.get_colmap_feature_cmd("/tmp/test.db", "/tmp/images")
        if len(test_cmd) > 0:
            print("  ✅ 配置方法正常工作")
            return True
        else:
            print("  ❌ 配置方法異常")
            return False
            
    except ImportError as e:
        print(f"  ❌ 配置文件導入失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🔍 COLMAP到NeRF流水線系統測試")
    print("=" * 50)
    
    tests = [
        ("Python版本", test_python_version),
        ("Python包", test_python_packages),
        ("COLMAP", test_colmap),
        ("目錄結構", test_directory_structure),
        ("原始圖片", test_raw_images),
        ("配置文件", test_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ 測試失敗: {e}")
            results.append((test_name, False))
    
    # 總結
    print("\n" + "=" * 50)
    print("📊 測試結果總結:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n通過率: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
    
    if passed == len(results):
        print("\n🎉 所有測試通過！系統已準備就緒。")
        print("💡 運行流水線: ./run_pipeline.sh 或 python colmap_nerf_pipeline.py --raw_images raw_images --project camper_nerf --verbose")
    else:
        print("\n⚠️ 部分測試失敗，請檢查上述問題。")
        
        # 提供安裝建議
        if not any(result for name, result in results if name in ["Python包", "COLMAP"]):
            print("\n📋 安裝建議:")
            print("  1. 安裝Python依賴: pip install -r requirements.txt")
            print("  2. 安裝COLMAP: https://colmap.github.io/install.html")

if __name__ == "__main__":
    main() 
# COLMAP到NeRF自動化流水線

這是一個完整的從原始圖片到NeRF格式的自動化處理系統，整合了COLMAP 3D重建和NeRF數據格式轉換。

## 🎯 系統概述

本系統提供了一個端到端的解決方案，將原始圖片自動處理成NeRF訓練所需的格式。整個流程包括：

1. **圖片預處理**: 將原始圖片複製到工作目錄
2. **COLMAP重建**: 自動執行特徵提取、匹配和稀疏重建
3. **格式轉換**: 將COLMAP輸出轉換為NeRF標準格式
4. **數據驗證**: 提供工具驗證轉換結果的質量

## 📁 目錄結構

```
camper_nerf_project/
├── README.md                    # 本文檔
├── colmap_nerf_pipeline.py      # 主要自動化腳本
├── colmap2nerf_fixed.py         # 修正版格式轉換工具
├── config.py                    # 配置文件
├── run_pipeline.sh              # 快速啟動腳本
├── test_setup.py                # 系統測試腳本
├── requirements.txt             # Python依賴
├── raw_images/                  # 原始圖片目錄
│   ├── Screenshot 2025-05-27 at 19.05.11.png
│   └── ...
├── camper_nerf/                 # 主要工作目錄
│   ├── images/                  # 處理用圖片 (從raw_images複製)
│   ├── colmap_output/           # COLMAP輸出
│   │   ├── database.db          # COLMAP數據庫
│   │   ├── sparse/              # 稀疏重建結果
│   │   │   └── 0/
│   │   │       ├── cameras.bin
│   │   │       ├── images.bin
│   │   │       └── points3D.bin
│   │   └── dense/               # 密集重建結果 (可選)
│   └── nerf_data/               # NeRF格式數據
│       └── transforms.json      # NeRF相機參數文件
└── data_validation/             # 數據驗證工具
    ├── app.py                   # Web驗證界面
    ├── validators/              # 驗證模塊
    └── templates/               # Web模板
```

## 🚀 快速開始

### 前置要求

1. **COLMAP**: 確保系統已安裝COLMAP
   ```bash
   # Ubuntu/Debian
   sudo apt install colmap
   
   # macOS
   brew install colmap
   
   # 或從源碼編譯: https://colmap.github.io/install.html
   ```

2. **Python依賴**:
   ```bash
   pip install -r requirements.txt
   ```

### 系統測試

在開始之前，建議先運行系統測試：

```bash
python test_setup.py
```

這會檢查所有依賴和配置是否正確。

### 基本使用

1. **準備圖片**: 將原始圖片放入 `raw_images/` 目錄

2. **運行完整流水線**:
   ```bash
   # 方法1: 使用快速啟動腳本 (推薦)
   ./run_pipeline.sh
   
   # 方法2: 直接使用Python腳本
   python colmap_nerf_pipeline.py --raw_images raw_images --project camper_nerf --verbose
   ```

3. **驗證結果**:
   ```bash
   cd data_validation
   python app.py
   ```

## 📋 詳細流程說明

### 步驟1: 圖片預處理
- 自動掃描 `raw_images/` 目錄中的圖片文件
- 支持格式: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`
- 複製到工作目錄 `camper_nerf/images/`
- 清理目標目錄確保乾淨的工作環境

### 步驟2: COLMAP 3D重建
執行完整的COLMAP流水線：

#### 2.1 特徵提取
```bash
colmap feature_extractor \
  --database_path camper_nerf/colmap_output/database.db \
  --image_path camper_nerf/images/ \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model PINHOLE \
  --SiftExtraction.use_gpu 1
```

#### 2.2 特徵匹配
```bash
colmap exhaustive_matcher \
  --database_path camper_nerf/colmap_output/database.db \
  --SiftMatching.use_gpu 1
```

#### 2.3 稀疏重建
```bash
colmap mapper \
  --database_path camper_nerf/colmap_output/database.db \
  --image_path camper_nerf/images/ \
  --output_path camper_nerf/colmap_output/sparse
```

### 步驟3: 格式轉換
使用修正版轉換工具 `colmap2nerf_fixed.py`：
- 讀取COLMAP的 `cameras.bin` 和 `images.bin`
- 修正座標系轉換 (COLMAP → NeRF)
- 標準化相機姿態
- 生成 `transforms.json` 文件

### 步驟4: 數據驗證
提供多種驗證工具：
- Web界面可視化相機軌跡
- 檢查相機姿態合理性
- 驗證圖片覆蓋範圍
- 分析重建質量

## 🛠️ 高級使用

### 快速啟動腳本選項

```bash
./run_pipeline.sh [選項]

選項:
  -r, --raw-images DIR     原始圖片目錄 (默認: raw_images)
  -p, --project DIR        項目工作目錄 (默認: camper_nerf)
  -s, --skip-colmap        跳過COLMAP處理，僅轉換現有結果
  -e, --env-config         使用環境變量配置
  -q, --quiet              靜默模式，不顯示詳細信息
  -h, --help               顯示幫助信息
```

### Python腳本選項

```bash
python colmap_nerf_pipeline.py [選項]

必需參數:
  --raw_images PATH     原始圖片目錄路徑
  --project PATH        項目工作目錄路徑

可選參數:
  --verbose            顯示詳細執行信息
  --skip-colmap        跳過COLMAP處理，僅轉換現有結果
  --config-from-env    從環境變量加載配置
  -h, --help           顯示幫助信息
```

### 使用示例

1. **標準流程**:
   ```bash
   ./run_pipeline.sh
   # 或
   python colmap_nerf_pipeline.py --raw_images raw_images --project camper_nerf --verbose
   ```

2. **僅轉換現有COLMAP結果**:
   ```bash
   ./run_pipeline.sh --skip-colmap
   # 或
   python colmap_nerf_pipeline.py --raw_images raw_images --project camper_nerf --skip-colmap
   ```

3. **使用自定義目錄**:
   ```bash
   ./run_pipeline.sh --raw-images my_images --project my_project
   ```

4. **使用環境變量配置**:
   ```bash
   export COLMAP_USE_GPU=0
   export COLMAP_MAX_IMAGE_SIZE=2048
   ./run_pipeline.sh --env-config
   ```

## 🔧 配置和調優

### COLMAP參數調整

如需調整COLMAP參數，可修改 `config.py` 中的相關配置：

```python
# 特徵提取參數
'feature_extractor': {
    'ImageReader.single_camera': '1',           # 假設單一相機
    'ImageReader.camera_model': 'PINHOLE',      # 相機模型
    'SiftExtraction.use_gpu': '1',              # 使用GPU加速
    'SiftExtraction.max_image_size': '3200',    # 最大圖片尺寸
    'SiftExtraction.max_num_features': '8192'   # 最大特徵點數
}
```

### 環境變量配置

支持以下環境變量：

```bash
export COLMAP_USE_GPU=0          # 禁用GPU加速
export COLMAP_MAX_IMAGE_SIZE=2048 # 設置最大圖片尺寸
export COLMAP_MAX_FEATURES=4096   # 設置最大特徵點數
```

### 座標系轉換

本系統使用修正版的座標系轉換，確保COLMAP和NeRF之間的正確對應：
- **COLMAP座標系**: X右, Y下, Z前
- **NeRF座標系**: X右, Y上, Z後

## 📊 輸出格式

### transforms.json結構
```json
{
  "camera_angle_x": 0.8575560450553894,
  "frames": [
    {
      "file_path": "IMG_0001.png",
      "rotation": 0.0,
      "transform_matrix": [
        [0.9998, -0.0123, 0.0156, 0.0234],
        [0.0125, 0.9999, -0.0089, -0.0167],
        [-0.0154, 0.0092, 0.9998, 0.9876],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ]
}
```

## 🔍 故障排除

### 常見問題

1. **COLMAP未安裝或不在PATH中**
   ```
   錯誤: FileNotFoundError: [Errno 2] No such file or directory: 'colmap'
   解決: 確保COLMAP已正確安裝並在系統PATH中
   ```

2. **GPU支持問題**
   ```
   錯誤: CUDA/OpenGL相關錯誤
   解決: 將GPU參數設為0或安裝適當的GPU驅動
   ```

3. **圖片格式不支持**
   ```
   錯誤: 沒有找到有效的圖片文件
   解決: 確保圖片格式為支持的類型，檢查文件擴展名
   ```

4. **COLMAP重建失敗**
   ```
   錯誤: 沒有生成cameras.bin
   解決: 檢查圖片質量、數量和重疊度，可能需要更多圖片或調整參數
   ```

### 調試技巧

1. **運行系統測試**:
   ```bash
   python test_setup.py
   ```

2. **使用詳細模式**:
   ```bash
   ./run_pipeline.sh --verbose
   ```

3. **檢查中間結果**:
   - 查看 `camper_nerf/colmap_output/database.db` 是否生成
   - 確認 `camper_nerf/colmap_output/sparse/0/` 中有三個.bin文件
   - 驗證 `camper_nerf/nerf_data/transforms.json` 格式正確

4. **分步執行**:
   ```bash
   # 先運行完整流程
   ./run_pipeline.sh
   
   # 如果COLMAP成功但轉換失敗，可以單獨重新轉換
   ./run_pipeline.sh --skip-colmap
   ```

## 🎯 最佳實踐

### 圖片拍攝建議
1. **數量**: 至少20-50張圖片，更多更好
2. **重疊度**: 相鄰圖片應有60-80%重疊
3. **角度**: 從多個角度拍攝，避免過於相似的視角
4. **質量**: 避免模糊、過曝或欠曝的圖片
5. **穩定性**: 確保場景在拍攝過程中沒有移動

### 工作流程建議
1. **先小規模測試**: 用少量圖片測試流程
2. **檢查中間結果**: 每步完成後檢查輸出
3. **使用驗證工具**: 運行數據驗證確保質量
4. **保留原始數據**: 不要刪除COLMAP中間文件，便於調試

## ⚡ 快速參考

### 完整工作流程
```bash
# 1. 系統測試
python test_setup.py

# 2. 運行流水線
./run_pipeline.sh

# 3. 驗證結果
cd data_validation && python app.py
```

### 常用命令
```bash
# 基本運行
./run_pipeline.sh

# 自定義目錄
./run_pipeline.sh -r my_images -p my_project

# 僅轉換
./run_pipeline.sh -s

# 禁用GPU
COLMAP_USE_GPU=0 ./run_pipeline.sh -e

# 靜默模式
./run_pipeline.sh -q
```

## 📚 相關資源

- [COLMAP官方文檔](https://colmap.github.io/)
- [NeRF論文](https://arxiv.org/abs/2003.08934)
- [Instant-NGP](https://github.com/NVlabs/instant-ngp)
- [NeRF-Studio](https://docs.nerf.studio/)

## 🤝 貢獻

歡迎提交Issue和Pull Request來改進這個工具！

## 📄 許可證

本項目採用MIT許可證。 
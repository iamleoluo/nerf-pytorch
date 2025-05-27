# NeRF 分析工具集

這個目錄包含了用於分析和檢測NeRF數據集的各種工具。

## 📁 目錄結構

```
tools/
├── analysis/           # 分析工具
│   ├── check_coordinate_conversion.py    # 座標轉換檢查工具
│   └── visualize_cameras.py             # 相機視覺化工具
└── README.md          # 本文件
```

## 🔧 工具說明

### 1. 座標轉換檢查工具 (`analysis/check_coordinate_conversion.py`)

**功能**: 詳細分析COLMAP到NeRF的座標系轉換是否正確

**主要特性**:
- 詳細說明COLMAP和NeRF座標系差異
- 分析當前轉換公式
- 用具體例子測試轉換
- 檢查四元數到旋轉矩陣的轉換
- 分析相機到世界座標的轉換
- 提出修正的轉換方法
- 檢查數據集中的變換矩陣

**使用方法**:
```bash
cd tools/analysis
python check_coordinate_conversion.py
```

**輸出**:
- 座標系統詳細分析
- 轉換公式檢查結果
- 數據集變換矩陣驗證
- 修正建議

### 2. 相機視覺化工具 (`analysis/visualize_cameras.py`)

**功能**: 視覺化和分析NeRF數據集中的相機位置和朝向分佈

**主要特性**:
- 提取相機位置和朝向
- 分析相機分佈統計
- 3D視覺化相機位置和朝向
- 2D投影分析
- 生成數據集品質報告
- 自動保存分析圖表

**使用方法**:
```bash
cd tools/analysis
python visualize_cameras.py
```

**輸出**:
- 相機分佈統計報告
- 3D相機位置視覺化圖
- 2D投影分析圖
- 數據集品質評估
- 圖表保存在 `outputs/camera_analysis/`

## 📊 分析指標

### 相機分佈品質評估標準

1. **相機朝向多樣性**: 最大相似度 < 0.95 為良好
2. **相機朝向中心**: 平均朝向中心度 > 0.3 為良好  
3. **位置分佈**: 距離標準差 > 0.1 為良好

### 品質等級

- **優秀**: 所有3項指標都達標
- **良好**: 2項指標達標
- **需要改善**: 1項指標達標
- **品質較差**: 0項指標達標

## 🚀 快速開始

1. 確保已安裝必要的依賴:
```bash
pip install numpy matplotlib scipy
```

2. 運行座標轉換檢查:
```bash
cd tools/analysis
python check_coordinate_conversion.py
```

3. 運行相機視覺化分析:
```bash
cd tools/analysis  
python visualize_cameras.py
```

## 📝 自定義使用

### 分析不同的數據集

修改腳本中的數據集路徑:
```python
# 在 visualize_cameras.py 中
dataset_path = "../../data/nerf_synthetic/your_dataset/transforms.json"

# 在 check_coordinate_conversion.py 中  
check_dataset_transforms("../../data/nerf_synthetic/your_dataset/transforms.json")
```

### 調整視覺化參數

在 `visualize_cameras.py` 中可以調整:
- 箭頭長度縮放: `scale = 0.1`
- 圖表大小: `figsize=(12, 10)`
- 顏色和透明度: `alpha=0.7`

## 🔍 故障排除

### 常見問題

1. **找不到數據集文件**
   - 檢查路徑是否正確
   - 確保 `transforms.json` 文件存在

2. **matplotlib顯示問題**
   - 在服務器環境中可能需要設置: `plt.switch_backend('Agg')`
   - 或者只保存圖片不顯示: 註釋掉 `plt.show()`

3. **路徑問題**
   - 確保從正確的目錄運行腳本
   - 檢查相對路徑是否正確

## 📈 輸出文件

分析結果會保存在以下位置:
- `outputs/camera_analysis/cameras_3d.png` - 3D相機視覺化
- `outputs/camera_analysis/cameras_2d.png` - 2D投影分析

## 🤝 貢獻

歡迎添加新的分析工具到這個目錄！請確保:
1. 添加適當的文檔說明
2. 使用一致的代碼風格
3. 更新這個README文件 
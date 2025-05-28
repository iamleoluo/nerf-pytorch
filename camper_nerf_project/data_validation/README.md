# NeRF 數據驗證系統

這是一個用於驗證 NeRF 和 COLMAP 數據一致性的可視化工具。該系統可以幫助您：
- 可視化 COLMAP 和 NeRF 的相機姿態
- 對比兩種方法的相機位置差異
- 驗證數據的完整性和一致性
- 生成詳細的驗證報告

## 系統要求

- Python 3.8 或更高版本
- pip（Python 包管理器）
- 現代瀏覽器（支持 WebGL）

## 安裝步驟

1. 克隆或下載項目到本地：
```bash
git clone <repository-url>
cd camper_nerf_project
```

2. 創建並激活虛擬環境（可選但推薦）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安裝依賴包：
```bash
cd data_validation
pip install -r requirements.txt
```

## 目錄結構

確保您的項目目錄結構如下：
```
camper_nerf_project/
├── camper_nerf/
│   ├── colmap_output/    # COLMAP 輸出文件
│   └── nerf_data/        # NeRF 數據文件
│       └── transforms.json
├── raw_images/           # 原始圖像文件
└── data_validation/      # 驗證系統代碼
    ├── app.py
    ├── requirements.txt
    ├── static/
    │   ├── css/
    │   └── js/
    └── templates/
```

## 運行系統

1. 確保您已經完成了 COLMAP 處理並生成了相應的輸出文件。

2. 啟動 Flask 服務器：
```bash
cd data_validation
python app.py
```

3. 在瀏覽器中訪問：
```
http://localhost:5000
```

## 使用說明

1. **載入數據**
   - 點擊"載入數據"按鈕加載相機數據
   - 系統會自動讀取 COLMAP 和 NeRF 的相機姿態

2. **可視化控制**
   - 使用開關按鈕控制顯示/隱藏 COLMAP（藍色）和 NeRF（紅色）相機
   - 使用滑鼠進行視圖操作：
     - 左鍵拖動：旋轉視圖
     - 右鍵拖動：平移視圖
     - 滾輪：縮放視圖

3. **相機信息查看**
   - 點擊任意相機查看其詳細信息
   - 系統會顯示對應的圖像和位置信息

4. **數據驗證**
   - 點擊"驗證數據"按鈕執行驗證
   - 查看驗證結果和警告信息

5. **導出報告**
   - 點擊"導出報告"按鈕生成 JSON 格式的驗證報告

## 故障排除

1. **無法啟動服務器**
   - 確保所有依賴包已正確安裝
   - 檢查端口 5000 是否被占用
   - 查看終端輸出的錯誤信息

2. **無法載入數據**
   - 確認目錄結構是否正確
   - 檢查文件權限
   - 查看瀏覽器控制台的錯誤信息

3. **3D 可視化問題**
   - 確保瀏覽器支持 WebGL
   - 更新顯卡驅動
   - 嘗試使用不同的瀏覽器

## 注意事項

- 確保 COLMAP 輸出和 NeRF 數據的相機 ID 對應關係正確
- 圖像文件路徑應與 transforms.json 中的路徑一致
- 建議定期備份驗證報告

## 貢獻

歡迎提交問題報告和改進建議！

## 許可證

[添加您的許可證信息] 
#!/bin/bash

# COLMAP到NeRF自動化流水線 - 無頭模式專用腳本
# 適用於SSH連接、Docker容器或無圖形界面的服務器環境

set -e  # 遇到錯誤時退出

echo "🚀 COLMAP到NeRF自動化流水線 (無頭模式)"
echo "適用於SSH連接和無圖形界面環境"
echo "=" * 50

# 設置無頭模式環境變量
export QT_QPA_PLATFORM=offscreen
export DISPLAY=:99
export QT_LOGGING_RULES="qt.qpa.xcb.debug=false"
export COLMAP_HEADLESS=1

# 可選：禁用GPU以避免顯示問題
if [ "${COLMAP_FORCE_CPU:-0}" = "1" ]; then
    export COLMAP_USE_GPU=0
    echo "🔧 強制使用CPU模式"
fi

# 顯示環境配置
echo "🔧 無頭模式環境配置:"
echo "   QT_QPA_PLATFORM: $QT_QPA_PLATFORM"
echo "   DISPLAY: $DISPLAY"
echo "   COLMAP_USE_GPU: ${COLMAP_USE_GPU:-1}"
echo ""

# 調用原始腳本
exec ./run_pipeline.sh "$@" 
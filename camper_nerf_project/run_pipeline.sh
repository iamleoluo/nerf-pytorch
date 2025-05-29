#!/bin/bash

# COLMAP到NeRF自動化流水線快速啟動腳本
# 使用方法: ./run_pipeline.sh [選項]

set -e  # 遇到錯誤時退出

# 默認配置
RAW_IMAGES_DIR="raw_images"
PROJECT_DIR="camper_nerf"
VERBOSE=true
SKIP_COLMAP=false
USE_ENV_CONFIG=false

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 幫助信息
show_help() {
    echo "COLMAP到NeRF自動化流水線快速啟動腳本"
    echo ""
    echo "使用方法: $0 [選項]"
    echo ""
    echo "選項:"
    echo "  -r, --raw-images DIR     原始圖片目錄 (默認: raw_images)"
    echo "  -p, --project DIR        項目工作目錄 (默認: camper_nerf)"
    echo "  -s, --skip-colmap        跳過COLMAP處理，僅轉換現有結果"
    echo "  -e, --env-config         使用環境變量配置"
    echo "  -q, --quiet              靜默模式，不顯示詳細信息"
    echo "  -h, --help               顯示此幫助信息"
    echo ""
    echo "環境變量配置:"
    echo "  COLMAP_USE_GPU=0         禁用GPU加速"
    echo "  COLMAP_MAX_IMAGE_SIZE=N  設置最大圖片尺寸"
    echo "  COLMAP_MAX_FEATURES=N    設置最大特徵點數"
    echo ""
    echo "示例:"
    echo "  $0                       # 使用默認設置運行完整流水線"
    echo "  $0 -r my_images -p my_project  # 指定自定義目錄"
    echo "  $0 -s                    # 僅轉換現有COLMAP結果"
    echo "  $0 -e                    # 使用環境變量配置"
}

# 日誌函數
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 檢查依賴
check_dependencies() {
    log_info "檢查系統依賴..."
    
    # 檢查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安裝或不在PATH中"
        exit 1
    fi
    
    # 檢查COLMAP
    if ! command -v colmap &> /dev/null; then
        log_error "COLMAP 未安裝或不在PATH中"
        log_info "請參考 https://colmap.github.io/install.html 安裝COLMAP"
        exit 1
    fi
    
    # 檢查Python依賴
    python3 -c "import numpy, cv2, scipy" 2>/dev/null || {
        log_error "Python依賴缺失，請運行: pip install numpy opencv-python scipy"
        exit 1
    }
    
    log_success "依賴檢查通過"
}

# 檢查輸入目錄
check_input_directory() {
    if [ ! -d "$RAW_IMAGES_DIR" ]; then
        log_error "原始圖片目錄不存在: $RAW_IMAGES_DIR"
        exit 1
    fi
    
    # 檢查是否有圖片文件
    image_count=$(find "$RAW_IMAGES_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.tif" \) | wc -l)
    
    if [ "$image_count" -eq 0 ]; then
        log_error "在 $RAW_IMAGES_DIR 中沒有找到圖片文件"
        exit 1
    fi
    
    log_info "找到 $image_count 張圖片"
}

# 解析命令行參數
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--raw-images)
                RAW_IMAGES_DIR="$2"
                shift 2
                ;;
            -p|--project)
                PROJECT_DIR="$2"
                shift 2
                ;;
            -s|--skip-colmap)
                SKIP_COLMAP=true
                shift
                ;;
            -e|--env-config)
                USE_ENV_CONFIG=true
                shift
                ;;
            -q|--quiet)
                VERBOSE=false
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知選項: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 構建Python命令
build_python_command() {
    local cmd="python3 colmap_nerf_pipeline.py"
    cmd="$cmd --raw_images $RAW_IMAGES_DIR"
    cmd="$cmd --project $PROJECT_DIR"
    
    if [ "$VERBOSE" = true ]; then
        cmd="$cmd --verbose"
    fi
    
    if [ "$SKIP_COLMAP" = true ]; then
        cmd="$cmd --skip-colmap"
    fi
    
    if [ "$USE_ENV_CONFIG" = true ]; then
        cmd="$cmd --config-from-env"
    fi
    
    echo "$cmd"
}

# 主函數
main() {
    echo "🚀 COLMAP到NeRF自動化流水線"
    echo "================================"
    
    # 解析參數
    parse_arguments "$@"
    
    # 檢查依賴
    check_dependencies
    
    # 檢查輸入目錄
    check_input_directory
    
    # 顯示配置
    log_info "配置信息:"
    log_info "  原始圖片目錄: $RAW_IMAGES_DIR"
    log_info "  項目工作目錄: $PROJECT_DIR"
    log_info "  跳過COLMAP: $SKIP_COLMAP"
    log_info "  使用環境配置: $USE_ENV_CONFIG"
    log_info "  詳細模式: $VERBOSE"
    
    # 構建並執行命令
    python_cmd=$(build_python_command)
    log_info "執行命令: $python_cmd"
    
    echo ""
    log_info "開始執行流水線..."
    
    # 執行Python腳本
    if eval "$python_cmd"; then
        echo ""
        log_success "流水線執行完成！"
        log_info "輸出文件位於: $PROJECT_DIR/nerf_data/transforms.json"
        log_info "建議運行數據驗證: cd data_validation && python app.py"
    else
        echo ""
        log_error "流水線執行失敗"
        exit 1
    fi
}

# 執行主函數
main "$@" 
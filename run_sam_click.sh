#!/bin/bash

# 使用方法: 
#   ./run_sam.sh microwave "microwave door"
#   ./run_sam.sh microwave "washing machine door" 12
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sam3


if [ -z "$1" ] || [ -z "$2" ]; then
    echo "用法: $0 <object_name> <prompt> [start_frame]"
    echo "示例: $0 microwave \"washing machine door\""
    echo "示例: $0 microwave \"washing machine door\" 13"
    exit 1
fi

OBJECT_NAME="$1"
PROMPT="$2"
START_FRAME="$3"  # 可选参数

BASE_DIR="/home/cfy/cfy/ccc/gs_robot_world/src/gs_robot_world/assets/object_assets"
OBJECT_DIR="${BASE_DIR}/${OBJECT_NAME}/render_output"
IMAGES_DIR="${OBJECT_DIR}/images"
OUTPUT_DIR="${OBJECT_DIR}/sam_results"

# 检查images目录是否存在
if [ ! -d "$IMAGES_DIR" ]; then
    echo "错误: 图像目录不存在: $IMAGES_DIR"
    exit 1
fi

# 构建命令
CMD="python sam3_video_click.py $IMAGES_DIR -o $OUTPUT_DIR --prompt \"$PROMPT\""

# 如果提供了start_frame参数，则添加
if [ ! -z "$START_FRAME" ]; then
    CMD="$CMD --start-frame $START_FRAME"
fi

# 执行命令
echo "========================================"
echo "运行SAM分割"
echo "========================================"
echo "物体: $OBJECT_NAME"
echo "提示词: $PROMPT"
if [ ! -z "$START_FRAME" ]; then
    echo "起始帧: $START_FRAME"
fi
echo "输出目录: $OUTPUT_DIR"
echo ""

eval $CMD

echo ""
echo "========================================"
echo "完成！"
echo "========================================"
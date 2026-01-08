#!/bin/bash

# 使用方法: 
#   ./run_segment.sh washing area_weighted 0.8
#   ./run_segment.sh washing intersection
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate drawer_sdf

if [ -z "$1" ]; then
    echo "用法: $0 <object_name> [mode]"
    echo "示例: $0 washing"
    echo "示例: $0 washing area_weighted"
    echo "示例: $0 washing intersection"
    exit 1
fi

OBJECT_NAME="$1"
MODE="${2:-vote}"  # 默认vote模式
PROMPT_TEXT="${PROMPT_TEXT:-}"

BASE_DIR="/home/jiziheng/Music/IROS2026/gs2colmap/assets/object_assets"
OBJECT_DIR="${BASE_DIR}/${OBJECT_NAME}/render_output"
PLY_FILE="${BASE_DIR}/${OBJECT_NAME}/rec_3d/${OBJECT_NAME}/${OBJECT_NAME}.ply"
MASKS_DIR="${OBJECT_DIR}/sam_results/masks"
TRANSFORMS_FILE="${OBJECT_DIR}/transforms.json"
OUTPUT_FILE="${OBJECT_DIR}/${OBJECT_NAME}_seg.ply"

# 检查输入文件是否存在
if [ ! -f "$PLY_FILE" ]; then
    echo "错误: PLY文件不存在: $PLY_FILE"
    exit 1
fi

if [ ! -d "$MASKS_DIR" ]; then
    echo "错误: Masks目录不存在: $MASKS_DIR"
    exit 1
fi

if [ ! -f "$TRANSFORMS_FILE" ]; then
    echo "错误: Transforms文件不存在: $TRANSFORMS_FILE"
    exit 1
fi

# 构建命令
CMD="python segment_gaussian.py \
    --ply $PLY_FILE \
    --masks $MASKS_DIR \
    --transforms $TRANSFORMS_FILE \
    --output $OUTPUT_FILE \
    --mode $MODE \
    --prompt \"$PROMPT_TEXT\" \
    --save-inverse \
    --core-threshold 0.5 \
    --restore-attributes"

# 执行命令
echo "========================================"
echo "分割Gaussian点云"
echo "========================================"
echo "物体: $OBJECT_NAME"
echo "模式: $MODE"
echo "输出文件: $OUTPUT_FILE"
echo ""

eval $CMD

echo ""
echo "========================================"
echo "完成！"
echo "========================================"

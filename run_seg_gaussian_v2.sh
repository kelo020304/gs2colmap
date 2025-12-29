#!/bin/bash

# 使用方法: 
#   ./run_segment.sh washing 0.5
#   ./run_segment.sh washing 0.7 --visualize

if [ -z "$1" ]; then
    echo "用法: $0 <object_name> [vote_threshold] [额外参数]"
    echo ""
    echo "示例:"
    echo "  $0 washing                    # 使用默认阈值0.5"
    echo "  $0 washing 0.7                # 使用阈值0.7"
    echo "  $0 washing 0.5 --visualize    # 可视化结果"
    echo "  $0 washing 0.6 --save-background  # 同时保存背景"
    exit 1
fi

OBJECT_NAME="$1"
VOTE_THRESHOLD="${2:-0.5}"  # 默认0.5
EXTRA_ARGS="${@:3}"  # 第3个参数开始的所有额外参数

BASE_DIR="/home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/ply_data"
OBJECT_DIR="${BASE_DIR}/${OBJECT_NAME}"
PLY_FILE="${OBJECT_DIR}/${OBJECT_NAME}.ply"
MASKS_DIR="${OBJECT_DIR}/sam_results/masks"
TRANSFORMS_FILE="${OBJECT_DIR}/transforms.json"
OUTPUT_FILE="${OBJECT_DIR}/${OBJECT_NAME}_seg.ply"

# 检查输入文件是否存在
if [ ! -f "$PLY_FILE" ]; then
    echo "❌ PLY文件不存在: $PLY_FILE"
    exit 1
fi

if [ ! -d "$MASKS_DIR" ]; then
    echo "❌ Masks目录不存在: $MASKS_DIR"
    exit 1
fi

if [ ! -f "$TRANSFORMS_FILE" ]; then
    echo "❌ Transforms文件不存在: $TRANSFORMS_FILE"
    exit 1
fi

# 构建命令
CMD="python gs2colmap/segment_gaussian_v2.py \
    --ply $PLY_FILE \
    --masks $MASKS_DIR \
    --transforms $TRANSFORMS_FILE \
    --output $OUTPUT_FILE \
    --vote-threshold $VOTE_THRESHOLD \
    --connectivity-radius 0.02 \
    --cluster-eps 0.02 \
    --cluster-min-samples 10 \
    $EXTRA_ARGS"

# 显示信息
echo ""
echo "========================================"
echo "🔧 3D Gaussian 点云分割"
echo "========================================"
echo "物体:       $OBJECT_NAME"
echo "投票阈值:   $VOTE_THRESHOLD"
echo "输入:"
echo "  PLY:      $PLY_FILE"
echo "  Masks:    $MASKS_DIR"
echo "输出:"
echo "  主文件:   $OUTPUT_FILE"
if [[ "$EXTRA_ARGS" == *"--save-background"* ]]; then
    echo "  背景:     ${OBJECT_DIR}/${OBJECT_NAME}_seg_background.ply"
fi
echo ""
echo "参数:"
echo "  连通半径: 0.02m (2cm)"
echo "  聚类半径: 0.02m (2cm)"
echo "  最小样本: 10"
echo ""

# 执行命令
eval $CMD

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "✅ 完成！"
    echo "========================================"
    echo ""
    echo "输出文件:"
    echo "  $OUTPUT_FILE"
    if [[ "$EXTRA_ARGS" == *"--save-background"* ]]; then
        echo "  ${OBJECT_DIR}/${OBJECT_NAME}_seg_background.ply"
    fi
    echo ""
    echo "💡 提示:"
    echo "  - 如果选中的点太少，降低阈值: ./run_segment.sh $OBJECT_NAME 0.3"
    echo "  - 如果有背景噪声，提高阈值:   ./run_segment.sh $OBJECT_NAME 0.7"
    echo "  - 可视化结果:                 ./run_segment.sh $OBJECT_NAME $VOTE_THRESHOLD --visualize"
else
    echo "========================================"
    echo "❌ 失败！退出码: $EXIT_CODE"
    echo "========================================"
fi

echo ""
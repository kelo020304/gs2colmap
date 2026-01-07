#!/bin/bash

if [ -z "$1" ]; then
    echo "用法: $0 <object_name>"
    echo "示例: $0 drawer_cabinet"
    exit 1
fi

OBJECT_NAME="$1"
BASE_DIR="/home/jiziheng/Music/IROS2026/gs2colmap/assets/object_assets"

TARGET_PLY="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_seg_mask0_gs.ply"
SOURCE_PLY="${BASE_DIR}/${OBJECT_NAME}/rec_3d/sub_drawers/drawer_0/drawer_0.ply"
OUTPUT_PLY="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_0_aligned.ply"

if [ ! -f "$SOURCE_PLY" ]; then
    echo "错误: SOURCE PLY不存在: $SOURCE_PLY"
    exit 1
fi

if [ ! -f "$TARGET_PLY" ]; then
    echo "错误: TARGET PLY不存在: $TARGET_PLY"
    exit 1
fi

python align_ply.py \
    --source "$SOURCE_PLY" \
    --target "$TARGET_PLY" \
    --output "$OUTPUT_PLY" \
    --restore-attributes

echo "✅ 完成: $OUTPUT_PLY"

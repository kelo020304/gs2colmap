#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "用法: $0 <object_name> <mask_id>"
    echo "示例: $0 drawer_cabinet 0"
    exit 1
fi

OBJECT_NAME="$1"
MASK_ID="$2"
BASE_DIR="/home/jiziheng/Music/IROS2026/gs2colmap/assets/object_assets"
MESH_DIR="${BASE_DIR}/${OBJECT_NAME}/rec_3d/${OBJECT_NAME}"
MESH_PATH="${MESH_DIR}/${OBJECT_NAME}.obj"
MASK_PLY="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_seg_mask${MASK_ID}_gs.ply"
OUT_PATH="${MESH_DIR}/${OBJECT_NAME}_seg_mask${MASK_ID}.obj"

if [ ! -f "$MESH_PATH" ]; then
    echo "错误: Mesh不存在: $MESH_PATH"
    exit 1
fi

if [ ! -f "$MASK_PLY" ]; then
    echo "错误: Mask PLY不存在: $MASK_PLY"
    exit 1
fi

python segment_mesh_by_ply.py \
    --mesh "$MESH_PATH" \
    --mask-ply "$MASK_PLY" \
    --output "$OUT_PATH" \
    --dist 0.006 \
    --min-keep 1

echo "✅ 完成: $OUT_PATH"

#!/bin/bash

if [ -z "$1" ]; then
    echo "用法: $0 <object_name> [object_id]"
    echo "示例: $0 drawer_cabinet"
    echo "示例: $0 drawer_cabinet 0"
    exit 1
fi

OBJECT_NAME="$1"
OBJECT_ID="${2:-}"
BASE_DIR="/home/jiziheng/Music/IROS2026/gs2colmap/assets/object_assets"
SUB_DRAWERS_DIR="${BASE_DIR}/${OBJECT_NAME}/rec_3d/sub_drawers"

IDS=()
if [ -n "$OBJECT_ID" ]; then
    IDS+=("$OBJECT_ID")
else
    if [ ! -d "$SUB_DRAWERS_DIR" ]; then
        echo "错误: 未找到 ${SUB_DRAWERS_DIR}"
        exit 1
    fi
    for d in "$SUB_DRAWERS_DIR"/drawer_*; do
        [ -d "$d" ] || continue
        id="${d##*/drawer_}"
        IDS+=("$id")
    done
    if [ ${#IDS[@]} -eq 0 ]; then
        echo "错误: 未找到任何 drawer_* 子目录"
        exit 1
    fi
fi

for OBJECT_ID in "${IDS[@]}"; do
    TARGET_PLY="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_seg_mask${OBJECT_ID}_gs.ply"
    SOURCE_PLY="${BASE_DIR}/${OBJECT_NAME}/rec_3d/sub_drawers/drawer_${OBJECT_ID}/drawer_${OBJECT_ID}.ply"
    OUTPUT_PLY="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_${OBJECT_ID}_aligned.ply"
    OUTPUT_JSON="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_${OBJECT_ID}_aligned.json"

    if [ ! -f "$SOURCE_PLY" ]; then
        echo "⚠️ SOURCE PLY不存在: $SOURCE_PLY"
        continue
    fi

    if [ ! -f "$TARGET_PLY" ]; then
        echo "⚠️ TARGET PLY不存在: $TARGET_PLY"
        continue
    fi

    python align_ply.py \
        --source "$SOURCE_PLY" \
        --target "$TARGET_PLY" \
        --output "$OUTPUT_PLY" \
        --output-json "$OUTPUT_JSON" \
        --align-obj

    TARGET_OBJ="${TARGET_PLY%.ply}.obj"
    TARGET_PLY_RAW="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_seg_mask${OBJECT_ID}.ply"
    TARGET_OBJ_RAW="${TARGET_PLY_RAW%.ply}.obj"
    rm -f "$TARGET_PLY" "$TARGET_OBJ" "$TARGET_PLY_RAW" "$TARGET_OBJ_RAW"

    BACKGROUND_PLY="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_seg_background_gs.ply"
    if [ ! -f "$BACKGROUND_PLY" ]; then
        BACKGROUND_PLY="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_seg_background.ply"
    fi

    if [ -f "$BACKGROUND_PLY" ]; then
        CLEAN_BG_PLY="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_seg_background_clean.ply"
        BACKGROUND_OBJ="${BACKGROUND_PLY%.ply}.obj"
        if [ ! -f "$BACKGROUND_OBJ" ]; then
            BACKGROUND_OBJ="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_seg_background.obj"
        fi
        CLEAN_BG_OBJ="${CLEAN_BG_PLY%.ply}.obj"

        if [ -f "$BACKGROUND_OBJ" ]; then
            python cleanup_background_by_ply.py \
                --background-ply "$BACKGROUND_PLY" \
                --drawer-ply "$OUTPUT_PLY" \
                --output-ply "$CLEAN_BG_PLY" \
                --background-obj "$BACKGROUND_OBJ" \
                --output-obj "$CLEAN_BG_OBJ"
        else
            python cleanup_background_by_ply.py \
                --background-ply "$BACKGROUND_PLY" \
                --drawer-ply "$OUTPUT_PLY" \
                --output-ply "$CLEAN_BG_PLY"
        fi
    else
        echo "⚠️ 未找到background PLY，跳过清理"
    fi

    echo "✅ 完成: $OUTPUT_PLY"
done

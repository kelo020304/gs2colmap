#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "用法: $0 <sam_prompt> [object_name]"
    echo "示例: $0 \"all drawers\""
    echo "示例: $0 \"all drawers\" drawer_cabinet"
    exit 1
fi

PROMPT="$1"
OBJECT_NAME="${2:-drawer_cabinet}"

echo "========================================"
echo "全自动流水线"
echo "物体: $OBJECT_NAME"
echo "提示词: $PROMPT"
echo "========================================"
echo ""

echo "Step 1: select_render"
OBJECT_NAME="$OBJECT_NAME" PROMPT_TEXT="$PROMPT" bash run_select_render.sh

echo ""
echo "Step 2: sam_click"
bash run_sam_click.sh "$OBJECT_NAME" "$PROMPT"

echo ""
echo "Step 3: seg_gaussian"
PROMPT_TEXT="$PROMPT" bash run_seg_gaussian.sh "$OBJECT_NAME"

echo ""
echo "Step 4: align + clean"
SUB_DRAWERS_DIR="/home/jiziheng/Music/IROS2026/gs2colmap/assets/object_assets/${OBJECT_NAME}/rec_3d/sub_items"
if [ -d "$SUB_DRAWERS_DIR" ] && ls -1 "$SUB_DRAWERS_DIR"/drawer_* >/dev/null 2>&1; then
    bash run_align_ply.sh "$OBJECT_NAME"
else
    echo "⚠️ 未找到sub_drawers，跳过align，执行clean"
    BASE_DIR="/home/jiziheng/Music/IROS2026/gs2colmap/assets/object_assets"
    BACKGROUND_PLY="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_seg_background_gs.ply"
    if [ ! -f "$BACKGROUND_PLY" ]; then
        BACKGROUND_PLY="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_seg_background.ply"
    fi
    DRAWER_PLY="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_seg_gs.ply"
    if [ ! -f "$DRAWER_PLY" ]; then
        DRAWER_PLY="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_seg.ply"
    fi
    if [ -f "$BACKGROUND_PLY" ] && [ -f "$DRAWER_PLY" ]; then
        CLEAN_BG_PLY="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_seg_background_clean.ply"
        BACKGROUND_OBJ="${BACKGROUND_PLY%.ply}.obj"
        if [ ! -f "$BACKGROUND_OBJ" ]; then
            BACKGROUND_OBJ="${BASE_DIR}/${OBJECT_NAME}/render_output/${OBJECT_NAME}_seg_background.obj"
        fi
        CLEAN_BG_OBJ="${CLEAN_BG_PLY%.ply}.obj"
        if [ -f "$BACKGROUND_OBJ" ]; then
            python cleanup_background_by_ply.py \
                --background-ply "$BACKGROUND_PLY" \
                --drawer-ply "$DRAWER_PLY" \
                --output-ply "$CLEAN_BG_PLY" \
                --background-obj "$BACKGROUND_OBJ" \
                --output-obj "$CLEAN_BG_OBJ"
        else
            python cleanup_background_by_ply.py \
                --background-ply "$BACKGROUND_PLY" \
                --drawer-ply "$DRAWER_PLY" \
                --output-ply "$CLEAN_BG_PLY"
        fi
    else
        echo "⚠️ 缺少seg或background PLY，跳过clean"
    fi
fi

echo ""
echo "========================================"
echo "完成！"
echo "========================================"

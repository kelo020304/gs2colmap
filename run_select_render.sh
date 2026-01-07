#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate drawer_sdf
# 设置输入参数
OBJECT_NAME="drawer_cabinet"
BASE_DIR="/home/jiziheng/Music/IROS2026/gs2colmap/assets/object_assets"
OBJECT_DIR="${BASE_DIR}/${OBJECT_NAME}/rec_3d/${OBJECT_NAME}"
PLY_FILE="${OBJECT_DIR}/${OBJECT_NAME}.ply"
OUTPUT_FILE="${BASE_DIR}/${OBJECT_NAME}/render_output"
TRAJ_FILE="${BASE_DIR}/${OBJECT_NAME}/render_output/traj.json"

# 检查PLY文件是否存在
if [ ! -f "$PLY_FILE" ]; then
    echo "错误: PLY文件不存在: $PLY_FILE"
    exit 1
fi

# Step 1: 选择物体
echo "========================================"
echo "Step 1: 选择物体"
echo "========================================"
python select_object.py \
    --ply "$PLY_FILE" \
    --output "$TRAJ_FILE" \
    --auto-center \
    --no-vis \
    --elevation-min -25 \
    --elevation-max 25 \
    --elevation-bands 3

# 检查是否成功生成轨迹文件
if [ ! -f "$TRAJ_FILE" ]; then
    echo "错误: 轨迹文件生成失败"
    exit 1
fi

# Step 2: 渲染
echo ""
echo "========================================"
echo "Step 2: 渲染"
echo "========================================"
python render.py \
    --ply "$PLY_FILE" \
    --trajectory "$TRAJ_FILE" \
    --output "$OUTPUT_FILE" \
    --width 640 \
    --height 480 \
    --fovy 65.0

echo ""
echo "========================================"
echo "完成！"
echo "========================================"
echo "输出目录: $OBJECT_DIR"

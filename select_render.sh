#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate drawer_sdf
# 设置输入参数
OBJECT_NAME="chest_of_drawers"
BASE_DIR="/home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/ply_data"
OBJECT_DIR="${BASE_DIR}/${OBJECT_NAME}"
PLY_FILE="${OBJECT_DIR}/${OBJECT_NAME}.ply"
TRAJ_FILE="${OBJECT_DIR}/traj.json"

# 检查PLY文件是否存在
if [ ! -f "$PLY_FILE" ]; then
    echo "错误: PLY文件不存在: $PLY_FILE"
    exit 1
fi

# Step 1: 选择物体
echo "========================================"
echo "Step 1: 选择物体"
echo "========================================"
python gs2colmap/select_object.py \
    --ply "$PLY_FILE" \
    --output "$TRAJ_FILE"

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
python gs2colmap/render.py \
    --ply "$PLY_FILE" \
    --trajectory "$TRAJ_FILE" \
    --output "$OBJECT_DIR" \
    --width 640 \
    --height 480 \
    --fovy 65.0

echo ""
echo "========================================"
echo "完成！"
echo "========================================"
echo "输出目录: $OBJECT_DIR"
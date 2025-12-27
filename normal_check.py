import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_normal_img(path):
    """
    读取法向图:
    - 假设为 uint8 图像 (0~255)
    - 转换为 float32 (0~1)
    - 再映射为 (-1~1)
    - 最后归一化（避免数值误差）
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    normal = img.astype(np.float32) / 255.0
    normal = normal * 2.0 - 1.0  # 映射到 [-1,1]

    # 归一化避免噪声
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / (norm + 1e-6)

    return normal

# =====================
# 读取两张法向图
# =====================
normal1 = load_normal_img("/home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/gs_ply/processed_data/normal/0000.png")
normal2 = load_normal_img("/home/jiziheng/Music/IROS2026/DRAWER/data/scene1_nerf/marigold_ft/normal_colored/frame_00001_pred_colored.png")

# =====================
# 计算差异图（点积 = 夹角余弦）
# =====================
dot = np.sum(normal1 * normal2, axis=-1)      # [-1,1]
dot_clamped = np.clip(dot, -1, 1)
angle = np.degrees(np.arccos(dot_clamped))    # 法向角度误差

# =====================
# 统计误差
# =====================
print("平均角度误差 (deg):", np.mean(angle))
print("最大角度误差 (deg):", np.max(angle))
print("中位数角度误差 (deg):", np.median(angle))

# =====================
# 可视化：两个法向图 + 误差热力图
# =====================
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Normal Map A")
plt.imshow((normal1 + 1) / 2)   # 映射回 0~1
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Normal Map B")
plt.imshow((normal2 + 1) / 2)
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Angle Error (deg)")
plt.imshow(angle, cmap='jet')
plt.colorbar()
plt.axis('off')

plt.show()

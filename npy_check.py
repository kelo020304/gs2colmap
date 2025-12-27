import numpy as np
import matplotlib.pyplot as plt

# 加载一个深度文件
depth = np.load("/home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/gs_ply/processed_data_1/depth/0078.npy")

print(f"Depth 统计:")
print(f"  Shape: {depth.shape}")
print(f"  Min: {depth.min():.6f}")
print(f"  Max: {depth.max():.6f}")
print(f"  Mean: {depth.mean():.6f}")
print(f"  Zero count: {(depth == 0).sum()} / {depth.size}")
print(f"  < 0.01: {(depth < 0.01).sum()} / {depth.size}")
print(f"  < 0.1: {(depth < 0.1).sum()} / {depth.size}")

# 可视化
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(depth, cmap='turbo')
plt.colorbar()
plt.title("原始深度")

plt.subplot(132)
plt.hist(depth.flatten(), bins=100)
plt.title("深度分布")
plt.xlabel("深度 (米)")
plt.yscale('log')

plt.subplot(133)
plt.imshow(depth == 0, cmap='gray')
plt.title("深度 < 0.1 的区域 (可能是无效)")

plt.tight_layout()
plt.savefig("depth_analysis.png", dpi=150)
print("保存到 depth_analysis.png")
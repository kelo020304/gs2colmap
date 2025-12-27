# 检查当前保存的深度
import numpy as np
import matplotlib.pyplot as plt

# 加载深度
depth = np.load("/home/jiziheng/Music/IROS2026/DRAWER/gs2colmap/gs_ply/processed_data_1/depth/0008.npy")

print(f"Depth shape: {depth.shape}")
print(f"Depth dtype: {depth.dtype}")
print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
print(f"Zero pixels: {(depth == 0).sum()} / {depth.size} ({(depth == 0).sum()/depth.size*100:.1f}%)")
print(f"Max depth pixels: {(depth == 10.0).sum()} (filled holes)")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ========== 修复 1: 原始深度 ==========
im0 = axes[0].imshow(depth, cmap='turbo', vmin=0, vmax=10)
axes[0].set_title("Saved Depth (NPY)")
plt.colorbar(im0, ax=axes[0])  # ← 修复: plt.colorbar()

# ========== 修复 2: 有效像素 ==========
valid_mask = (depth > 0) & (depth < 10)
axes[1].imshow(valid_mask, cmap='gray')
axes[1].set_title(f"Valid Pixels ({valid_mask.sum()/valid_mask.size*100:.1f}%)")

# ========== 修复 3: Masked 深度 ==========
masked_depth = depth.copy()
masked_depth[~valid_mask] = np.nan
im2 = axes[2].imshow(masked_depth, cmap='turbo', vmin=0, vmax=10)
axes[2].set_title("Valid Depth Only")
plt.colorbar(im2, ax=axes[2])  # ← 修复: plt.colorbar()

plt.tight_layout()
plt.savefig("depth_check.png", dpi=150, bbox_inches='tight')
print("\n✓ Saved depth_check.png")

# ========== 额外: 显示统计信息 ==========
print("\n" + "="*60)
print("深度统计:")
print("="*60)
print(f"有效像素: {valid_mask.sum()} ({valid_mask.sum()/valid_mask.size*100:.1f}%)")
print(f"空洞像素: {(depth == 0).sum()} ({(depth == 0).sum()/depth.size*100:.1f}%)")
print(f"有效深度范围: [{depth[valid_mask].min():.3f}, {depth[valid_mask].max():.3f}] 米")
print(f"平均深度: {depth[valid_mask].mean():.3f} 米")
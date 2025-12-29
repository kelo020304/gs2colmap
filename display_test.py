#!/usr/bin/env python3
"""
测试SAM3实际加载模型后的cv2.imshow
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import cv2
import numpy as np
import torch
from pathlib import Path
from sam3.model_builder import build_sam3_video_predictor

print("="*60)
print("SAM3 + cv2.imshow 段错误诊断")
print("="*60)

# 1. 加载SAM3
print("\n1. 加载SAM3...")
predictor = build_sam3_video_predictor()
print("✅ SAM3加载成功")

# 2. 创建session
print("\n2. 创建session...")
video_dir = "washine_machine"
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_dir,
    )
)
session_id = response["session_id"]
print(f"✅ Session: {session_id}")

# 3. 获取一帧
frames = sorted(Path(video_dir).glob("*.png"))
frame_idx = 264
print(f"\n3. 读取帧 {frame_idx}...")
img = cv2.imread(str(frames[frame_idx]))
print(f"   Shape: {img.shape}, dtype: {img.dtype}")

# 4. 测试直接显示原图
print("\n4. 测试显示原图...")
try:
    cv2.namedWindow("Test Original", cv2.WINDOW_NORMAL)
    cv2.imshow("Test Original", img)
    print("   ✅ 原图显示成功")
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
except Exception as e:
    print(f"   ❌ 原图显示失败: {e}")

# 5. 运行text prompt
print("\n5. 运行text prompt...")
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        text="washing machine door",
    )
)

outputs = response.get("outputs", {})
print(f"   Outputs keys: {outputs.keys()}")

# 6. 提取mask
if "out_binary_masks" in outputs:
    mask = outputs["out_binary_masks"][0]
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    while mask.ndim > 2:
        mask = mask[0]
    
    print(f"   Mask shape: {mask.shape}, dtype: {mask.dtype}")
    
    # Resize mask
    if mask.shape != (img.shape[0], img.shape[1]):
        print(f"   Resizing mask from {mask.shape} to {img.shape[:2]}")
        mask = cv2.resize(mask.astype(np.uint8), 
                         (img.shape[1], img.shape[0]), 
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        mask = mask > 0.5
    
    print(f"   Final mask shape: {mask.shape}, dtype: {mask.dtype}")
    
    # 7. 测试显示mask
    print("\n7. 测试显示mask...")
    try:
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.namedWindow("Test Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Test Mask", mask_uint8)
        print("   ✅ Mask显示成功")
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"   ❌ Mask显示失败: {e}")
    
    # 8. 创建overlay
    print("\n8. 测试overlay...")
    try:
        # 方法A: 直接索引
        print("   方法A: overlay[mask] = ...")
        overlay_a = img.copy().astype(np.float32)
        overlay_a[mask] = overlay_a[mask] * 0.6 + np.array([30, 255, 30]) * 0.4
        overlay_a = overlay_a.astype(np.uint8)
        print(f"   Overlay A shape: {overlay_a.shape}, dtype: {overlay_a.dtype}")
        
        # 检查是否是contiguous
        print(f"   Contiguous: {overlay_a.flags['C_CONTIGUOUS']}")
        
        cv2.namedWindow("Test Overlay A", cv2.WINDOW_NORMAL)
        cv2.imshow("Test Overlay A", overlay_a)
        print("   ✅ Overlay A显示成功")
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"   ❌ Overlay A显示失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 方法B: np.where
    try:
        print("\n   方法B: np.where...")
        overlay_b = img.copy().astype(np.float32)
        overlay_b[:,:,0] = np.where(mask, overlay_b[:,:,0] * 0.6 + 30 * 0.4, overlay_b[:,:,0])
        overlay_b[:,:,1] = np.where(mask, overlay_b[:,:,1] * 0.6 + 255 * 0.4, overlay_b[:,:,1])
        overlay_b[:,:,2] = np.where(mask, overlay_b[:,:,2] * 0.6 + 30 * 0.4, overlay_b[:,:,2])
        overlay_b = overlay_b.astype(np.uint8)
        print(f"   Overlay B shape: {overlay_b.shape}, dtype: {overlay_b.dtype}")
        print(f"   Contiguous: {overlay_b.flags['C_CONTIGUOUS']}")
        
        cv2.namedWindow("Test Overlay B", cv2.WINDOW_NORMAL)
        cv2.imshow("Test Overlay B", overlay_b)
        print("   ✅ Overlay B显示成功")
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"   ❌ Overlay B显示失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 9. 测试ascontiguousarray
    try:
        print("\n   方法C: ascontiguousarray...")
        overlay_c = img.copy().astype(np.float32)
        overlay_c[mask] = overlay_c[mask] * 0.6 + np.array([30, 255, 30]) * 0.4
        overlay_c = np.ascontiguousarray(overlay_c.astype(np.uint8))
        print(f"   Overlay C shape: {overlay_c.shape}, dtype: {overlay_c.dtype}")
        print(f"   Contiguous: {overlay_c.flags['C_CONTIGUOUS']}")
        
        cv2.namedWindow("Test Overlay C", cv2.WINDOW_NORMAL)
        cv2.imshow("Test Overlay C", overlay_c)
        print("   ✅ Overlay C显示成功")
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"   ❌ Overlay C显示失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 10. 测试循环显示
    print("\n10. 测试循环显示overlay...")
    try:
        overlay = np.ascontiguousarray(overlay_c.copy())
        
        cv2.namedWindow("Test Loop", cv2.WINDOW_NORMAL)
        
        for i in range(5):
            cv2.putText(overlay, f"Frame {i}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            print(f"   显示帧 {i}...")
            cv2.imshow("Test Loop", overlay)
            key = cv2.waitKey(500)
            
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("   ✅ 循环显示成功")
    except Exception as e:
        print(f"   ❌ 循环显示失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("诊断完成")
print("="*60)
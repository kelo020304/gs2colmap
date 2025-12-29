#!/usr/bin/env python3
"""
SAM3 Video Tracking - 使用官方notebook的点击交互方式
每次点击都重新生成mask，不使用复杂的refinement API
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Circle

from sam3.model_builder import build_sam3_video_predictor


def safe_overlay(img, mask):
    """安全地创建mask overlay"""
    overlay = img.copy().astype(np.float32)
    for c in range(3):
        if c == 1:
            overlay[:,:,c] = np.where(mask, overlay[:,:,c] * 0.6 + 255 * 0.4, overlay[:,:,c])
        else:
            overlay[:,:,c] = np.where(mask, overlay[:,:,c] * 0.6 + 30 * 0.4, overlay[:,:,c])
    return overlay.astype(np.uint8)


class ClickBasedMaskEditor:
    """点击式mask编辑器 - 每次点击都生成包围所有点的box来重新生成mask"""
    
    def __init__(self, predictor, session_id, frame_idx, image_path, text_prompt):
        self.predictor = predictor
        self.session_id = session_id
        self.frame_idx = frame_idx
        self.text_prompt = text_prompt
        self.img = cv2.imread(str(image_path))
        self.H, self.W = self.img.shape[:2]
        
        # 点击历史
        self.fg_points = []  # foreground points
        self.bg_points = []  # background points
        
        self.current_mask = None
        self.confirmed = False
        self.cancelled = False
        
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.canvas.manager.set_window_title('Click-based Mask Editor')
        
        self.update_display()
        
        # 按钮
        ax_done = plt.axes([0.7, 0.02, 0.08, 0.04])
        ax_reset = plt.axes([0.6, 0.02, 0.09, 0.04])
        ax_cancel = plt.axes([0.5, 0.02, 0.09, 0.04])
        
        self.btn_done = Button(ax_done, 'Done ✓', color='lightgreen')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_cancel = Button(ax_cancel, 'Cancel', color='lightcoral')
        
        self.btn_done.on_clicked(lambda e: self.on_done())
        self.btn_reset.on_clicked(lambda e: self.on_reset())
        self.btn_cancel.on_clicked(lambda e: self.on_cancel())
        
        # 事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        print("\n🖱️ Click-based Mask Editor:")
        print("   Left Click: Include this area (foreground)")
        print("   Right Click: Exclude this area (background)")
        print("   R: Reset all clicks")
        print("   Space/Done: Confirm")
        print(f"   Text prompt: '{text_prompt}'")
    
    def generate_initial_mask(self):
        """用text prompt生成初始mask"""
        try:
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=self.session_id,
                    frame_index=self.frame_idx,
                    text=self.text_prompt,
                )
            )
            
            outputs = response.get("outputs", {})
            
            if "out_binary_masks" in outputs and outputs["out_binary_masks"].shape[0] > 0:
                mask = outputs["out_binary_masks"][0]
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                if mask.shape != (self.H, self.W):
                    mask = cv2.resize(mask.astype(np.uint8), 
                                     (self.W, self.H), 
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    mask = mask > 0.5
                
                self.current_mask = mask
                print(f"   ✅ Initial mask generated")
            else:
                print(f"   ⚠️ No initial mask generated")
        
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
        
        self.update_display()
    
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        if event.button == 1:  # 左键 = foreground
            self.fg_points.append([x, y])
            print(f"   ✅ Foreground point {len(self.fg_points) + len(self.bg_points)} at ({x}, {y})")
        elif event.button == 3:  # 右键 = background
            self.bg_points.append([x, y])
            print(f"   ❌ Background point {len(self.fg_points) + len(self.bg_points)} at ({x}, {y})")
        else:
            return
        
        # 重新生成mask
        self.regenerate_mask()
    
    def regenerate_mask(self):
        """基于当前所有点重新生成mask"""
        if len(self.fg_points) == 0:
            # 没有foreground点，不生成mask
            self.current_mask = None
            self.update_display()
            return
        
        # 策略：用所有foreground点的包围盒作为positive box
        # 每个background点转换为一个小的negative box
        
        fg_points_np = np.array(self.fg_points, dtype=np.float32)
        
        # Foreground box: 包围所有fg points
        x_min = fg_points_np[:, 0].min()
        x_max = fg_points_np[:, 0].max()
        y_min = fg_points_np[:, 1].min()
        y_max = fg_points_np[:, 1].max()
        
        # 扩大一点
        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(self.W, x_max + margin)
        y_max = min(self.H, y_max + margin)
        
        # 转换为normalized cxcywh
        center_x = ((x_min + x_max) / 2) / self.W
        center_y = ((y_min + y_max) / 2) / self.H
        width = (x_max - x_min) / self.W
        height = (y_max - y_min) / self.H
        
        boxes = [[center_x, center_y, width, height]]
        box_labels = [1]  # positive
        
        # Background boxes: 每个bg点周围的小box
        for bg_point in self.bg_points:
            bx, by = bg_point
            # 小box（20x20像素）
            bg_size = 20
            bg_x_min = max(0, bx - bg_size/2)
            bg_y_min = max(0, by - bg_size/2)
            bg_x_max = min(self.W, bx + bg_size/2)
            bg_y_max = min(self.H, by + bg_size/2)
            
            bg_cx = ((bg_x_min + bg_x_max) / 2) / self.W
            bg_cy = ((bg_y_min + bg_y_max) / 2) / self.H
            bg_w = (bg_x_max - bg_x_min) / self.W
            bg_h = (bg_y_max - bg_y_min) / self.H
            
            boxes.append([bg_cx, bg_cy, bg_w, bg_h])
            box_labels.append(0)  # negative
        
        try:
            # 重新调用add_prompt (会reset之前的state)
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=self.session_id,
                    frame_index=self.frame_idx,
                    text=self.text_prompt,
                    boxes_xywh=boxes,
                    box_labels=box_labels,
                )
            )
            
            outputs = response.get("outputs", {})
            
            if "out_binary_masks" in outputs and outputs["out_binary_masks"].shape[0] > 0:
                mask = outputs["out_binary_masks"][0]
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                if mask.shape != (self.H, self.W):
                    mask = cv2.resize(mask.astype(np.uint8), 
                                     (self.W, self.H), 
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    mask = mask > 0.5
                
                self.current_mask = mask
                print(f"   ✅ Mask updated")
            else:
                print(f"   ⚠️ No mask generated")
        
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
        
        self.update_display()
    
    def update_display(self):
        """更新显示"""
        self.ax.clear()
        
        # 显示图像 + mask
        if self.current_mask is not None:
            overlay = safe_overlay(self.img, self.current_mask)
            self.ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        else:
            self.ax.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        
        # 显示点
        for x, y in self.fg_points:
            circle = Circle((x, y), 6, color='lime', fill=True, alpha=0.9, zorder=10)
            self.ax.add_patch(circle)
            self.ax.plot(x, y, '+', color='black', markersize=10, markeredgewidth=2, zorder=11)
        
        for x, y in self.bg_points:
            circle = Circle((x, y), 6, color='red', fill=True, alpha=0.9, zorder=10)
            self.ax.add_patch(circle)
            self.ax.plot(x, y, 'x', color='white', markersize=10, markeredgewidth=2, zorder=11)
        
        # 标题
        total = len(self.fg_points) + len(self.bg_points)
        if total == 0:
            if self.current_mask is not None:
                title = f'Initial mask from: "{self.text_prompt}"\nClick to refine, or press Space to accept'
            else:
                title = f'Text: "{self.text_prompt}"\nWaiting for initial mask...'
        else:
            title = f'Clicks: {total} ({len(self.fg_points)} include, {len(self.bg_points)} exclude)'
        
        self.ax.set_title(title, fontsize=12, pad=15)
        self.ax.axis('off')
        
        self.fig.canvas.draw()
    
    def on_reset(self):
        """重置"""
        self.fg_points.clear()
        self.bg_points.clear()
        self.current_mask = None
        print("   🔄 Reset all clicks")
        self.update_display()
    
    def on_key(self, event):
        if event.key == ' ' and self.current_mask is not None:
            self.on_done()
        elif event.key in ['r', 'R']:
            self.on_reset()
        elif event.key == 'escape':
            self.on_cancel()
    
    def on_done(self):
        if self.current_mask is None:
            print("   ⚠️ No mask to confirm!")
            return
        
        print("✅ Mask confirmed!")
        self.confirmed = True
        plt.close(self.fig)
    
    def on_cancel(self):
        print("❌ Cancelled")
        self.cancelled = True
        plt.close(self.fig)
    
    def show(self):
        plt.show()
        
        if self.cancelled:
            return None
        return self.current_mask


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=30)
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(exist_ok=True)
    (output_dir / "viz").mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🎬 SAM3 Video Tracking - Click-based")
    print(f"{'='*60}\n")
    
    print("Loading SAM3...")
    predictor = build_sam3_video_predictor()
    
    print(f"Starting session: {video_dir}")
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=str(video_dir))
    )
    session_id = response["session_id"]
    print(f"✅ Session: {session_id}\n")
    
    frames = sorted(video_dir.glob("*.png")) or sorted(video_dir.glob("*.jpg"))
    print(f"📹 Total frames: {len(frames)}")
    print(f"🎯 Prompt frame: {args.start_frame}\n")
    
    # 交互式点击编辑
    editor = ClickBasedMaskEditor(
        predictor,
        session_id,
        args.start_frame,
        frames[args.start_frame],
        args.prompt
    )
    
    # 先用text生成初始mask
    print(f"🎯 Generating initial mask with text: '{args.prompt}'...")
    editor.generate_initial_mask()
    
    # 显示并允许refine
    final_mask = editor.show()
    
    if final_mask is None:
        print("Cancelled")
        return
    
    # Propagate
    print(f"\n📹 Propagating...")
    
    video_segments = {}
    
    print(f"   → Forward")
    for frame_outputs in predictor.propagate_in_video(
        session_id=session_id,
        propagation_direction="forward",
        start_frame_idx=args.start_frame,
        max_frame_num_to_track=len(frames) - args.start_frame
    ):
        if isinstance(frame_outputs, dict):
            frame_idx = frame_outputs.get("frame_index")
            outputs_dict = frame_outputs.get("outputs", {})
            
            mask_logits = outputs_dict.get("out_mask_logits") or outputs_dict.get("out_binary_masks")
            
            if mask_logits is not None and len(mask_logits) > 0:
                mask = mask_logits[0]
                if isinstance(mask, torch.Tensor):
                    mask = (mask > 0.0).cpu().numpy()
                while mask.ndim > 2:
                    mask = mask[0]
                
                if frame_idx is not None:
                    video_segments[frame_idx] = mask > 0.5
    
    if args.start_frame > 0:
        print(f"   → Backward")
        for frame_outputs in predictor.propagate_in_video(
            session_id=session_id,
            propagation_direction="backward",
            start_frame_idx=args.start_frame,
            max_frame_num_to_track=args.start_frame + 1
        ):
            if isinstance(frame_outputs, dict):
                frame_idx = frame_outputs.get("frame_index")
                outputs_dict = frame_outputs.get("outputs", {})
                
                mask_logits = outputs_dict.get("out_mask_logits") or outputs_dict.get("out_binary_masks")
                
                if mask_logits is not None and len(mask_logits) > 0:
                    mask = mask_logits[0]
                    if isinstance(mask, torch.Tensor):
                        mask = (mask > 0.0).cpu().numpy()
                    while mask.ndim > 2:
                        mask = mask[0]
                    
                    if frame_idx is not None:
                        video_segments[frame_idx] = mask > 0.5
    
    print(f"✅ Tracked {len(video_segments)} frames\n")
    
    if len(video_segments) == 0:
        print("❌ No masks!")
        return
    
    print("💾 Saving...")
    
    saved_count = 0
    viz_frames = []
    
    for frame_idx in tqdm(sorted(video_segments.keys())):
        if frame_idx >= len(frames):
            continue
        
        try:
            mask = video_segments[frame_idx]
            
            img = cv2.imread(str(frames[frame_idx]))
            if img is None:
                continue
            
            if mask.shape != (img.shape[0], img.shape[1]):
                mask = cv2.resize(mask.astype(np.uint8), 
                                 (img.shape[1], img.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # 使用frame_idx作为文件名（与transforms.json的索引对应）
            mask_uint8 = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / "masks" / f"{frame_idx:04d}.png"), mask_uint8)
            
            overlay = safe_overlay(img, mask)
            cv2.putText(overlay, f"Frame {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imwrite(str(output_dir / "viz" / f"{frame_idx:04d}.png"), overlay)
            
            viz_frames.append(overlay)
            saved_count += 1
        
        except Exception as e:
            print(f"\n⚠️  Error frame {frame_idx}: {e}")
            continue
    
    if len(viz_frames) > 0:
        print(f"\n🎬 Creating video...")
        video_path = output_dir / "tracking_video.mp4"
        
        total_frames_needed = args.video_fps * 10
        
        looped_frames = []
        while len(looped_frames) < total_frames_needed:
            looped_frames.extend(viz_frames)
        looped_frames = looped_frames[:total_frames_needed]
        
        h, w = viz_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, args.video_fps, (w, h))
        
        for frame in tqdm(looped_frames, desc="Writing video"):
            video_writer.write(frame)
        
        video_writer.release()
        print(f"✅ Video: {video_path}")
    
    metadata = {
        "prompt": args.prompt,
        "start_frame": args.start_frame,
        "num_frames_tracked": len(video_segments),
        "num_frames_saved": saved_count,
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Done!")
    print(f"   Tracked: {len(video_segments)} frames")
    print(f"   Saved: {saved_count} frames")
    print(f"   Video: {output_dir / 'tracking_video.mp4'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
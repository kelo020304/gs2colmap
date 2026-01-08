#!/usr/bin/env python3
"""
SAM3 Video Tracking - 多对象分割版本
支持为不同区域分配不同的obj_id
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import cv2
import numpy as np
from pathlib import Path
import json
import re
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Circle, Rectangle

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


def _find_sub_drawers_mask_dir(video_dir):
    """从视频目录向上查找 sub_item_masks 目录"""
    for parent in [video_dir] + list(video_dir.parents):
        candidate = parent / "sub_item_masks"
        if candidate.is_dir():
            return candidate
    return None


def _drawer_mask_sort_key(path):
    """按文件名中的数字排序，保证 drawer_mask_0 对应 id=0"""
    match = re.search(r"(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return path.stem


def _mask_to_box_xywh(mask, target_h, target_w):
    """从二值mask计算归一化的xywh box"""
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None

    x_min = xs.min()
    x_max = xs.max() + 1
    y_min = ys.min()
    y_max = ys.max() + 1

    mask_h, mask_w = mask.shape[:2]
    if (mask_h, mask_w) != (target_h, target_w):
        scale_x = target_w / float(mask_w)
        scale_y = target_h / float(mask_h)
        x_min = int(round(x_min * scale_x))
        x_max = int(round(x_max * scale_x))
        y_min = int(round(y_min * scale_y))
        y_max = int(round(y_max * scale_y))

    x_min = max(0, min(x_min, target_w - 1))
    x_max = max(1, min(x_max, target_w))
    y_min = max(0, min(y_min, target_h - 1))
    y_max = max(1, min(y_max, target_h))

    center_x = ((x_min + x_max) / 2.0) / target_w
    center_y = ((y_min + y_max) / 2.0) / target_h
    width = (x_max - x_min) / target_w
    height = (y_max - y_min) / target_h

    return [center_x, center_y, width, height]


def _mask_centroid(mask):
    """计算mask质心坐标 (x, y)"""
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def _rescale_point(point, src_h, src_w, dst_h, dst_w):
    """按尺寸比例缩放点坐标"""
    x, y = point
    return (x * dst_w / float(src_w), y * dst_h / float(src_h))


def _load_reference_centroids(mask_dir, target_h, target_w):
    """加载参考mask质心，按文件名顺序返回"""
    mask_paths = sorted(mask_dir.glob("*.png"), key=_drawer_mask_sort_key)
    centroids = []
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"   ⚠️ Skip unreadable mask: {mask_path}")
            continue
        centroid = _mask_centroid(mask)
        if centroid is None:
            print(f"   ⚠️ Skip empty mask: {mask_path}")
            continue
        mask_h, mask_w = mask.shape[:2]
        if (mask_h, mask_w) != (target_h, target_w):
            centroid = _rescale_point(centroid, mask_h, mask_w, target_h, target_w)
        centroids.append((mask_path.name, centroid))
    return centroids


def _compute_greedy_mapping(ref_centroids, sam_centroids):
    """按最近距离做贪心匹配，返回 ref_id -> sam_id 的映射"""
    mapping = {}
    unused_sam = set(range(len(sam_centroids)))
    for ref_idx, (_, ref_pt) in enumerate(ref_centroids):
        if not unused_sam:
            break
        best_sam = None
        best_dist = None
        for sam_idx in unused_sam:
            sam_pt = sam_centroids[sam_idx]
            if sam_pt is None:
                continue
            dx = ref_pt[0] - sam_pt[0]
            dy = ref_pt[1] - sam_pt[1]
            dist = dx * dx + dy * dy
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_sam = sam_idx
        if best_sam is not None:
            mapping[ref_idx] = best_sam
            unused_sam.remove(best_sam)
    return mapping


def add_prompts_from_drawer_masks(predictor, session_id, frame_idx, frame_path, text_prompt, video_dir):
    """从 sub_item_masks 自动生成多对象 prompts"""
    mask_dir = _find_sub_drawers_mask_dir(video_dir)
    if mask_dir is None:
        return 0

    mask_paths = sorted(mask_dir.glob("*.png"), key=_drawer_mask_sort_key)
    if len(mask_paths) == 0:
        return 0

    img = cv2.imread(str(frame_path))
    if img is None:
        return 0
    target_h, target_w = img.shape[:2]

    print(f"🧩 Auto drawer masks: {mask_dir} ({len(mask_paths)} masks)")

    mask_infos = []
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"   ⚠️ Skip unreadable mask: {mask_path}")
            continue
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            print(f"   ⚠️ Skip empty mask: {mask_path}")
            continue
        mask_infos.append((mask_path, mask))

    if len(mask_infos) == 0:
        return 0

    # 严格按文件名顺序，保证 drawer_mask_0 -> object_id=0
    mask_infos.sort(key=lambda item: _drawer_mask_sort_key(item[0]))

    added = 0
    for obj_idx, (mask_path, mask) in enumerate(mask_infos):
        box_xywh = _mask_to_box_xywh(mask, target_h, target_w)
        if box_xywh is None:
            print(f"   ⚠️ Skip empty mask: {mask_path}")
            continue

        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_idx,
                object_id=obj_idx,
                text=text_prompt,
                boxes_xywh=[box_xywh],
                box_labels=[1],
            )
        )
        added += 1
        print(f"   ✅ Added drawer mask {mask_path.name} as object_id={obj_idx}")

    return added


def _extract_masks_from_outputs(frame_outputs):
    """兼容不同版本的propagate输出格式"""
    frame_idx = None
    outputs_dict = None
    mask_logits = None
    
    if isinstance(frame_outputs, dict):
        frame_idx = frame_outputs.get("frame_index")
        outputs_dict = frame_outputs.get("outputs", {})
    elif isinstance(frame_outputs, (list, tuple)) and len(frame_outputs) == 2:
        frame_idx = frame_outputs[0]
        outputs_dict = frame_outputs[1]
    else:
        return frame_idx, mask_logits
    
    if isinstance(outputs_dict, dict):
        mask_logits = outputs_dict.get("out_mask_logits")
        if mask_logits is None:
            mask_logits = outputs_dict.get("out_binary_masks")
        if mask_logits is None:
            mask_logits = outputs_dict.get("out_masks")
    else:
        mask_logits = outputs_dict
    
    return frame_idx, mask_logits


class MultiObjectMaskEditor:
    """多对象mask编辑器 - 每个prompt可以是独立的物体"""
    
    def __init__(self, predictor, session_id, frame_idx, image_path, text_prompt):
        self.predictor = predictor
        self.session_id = session_id
        self.frame_idx = frame_idx
        self.text_prompt = text_prompt
        self.img = cv2.imread(str(image_path))
        self.H, self.W = self.img.shape[:2]
        
        # 交互模式
        self.click_mode = "box"  # "point" 或 "box"
        self.point_mode = "positive"
        
        # 多对象支持
        self.objects = []  # [{"boxes": [...], "points": [...], "mask": ...}, ...]
        self.current_obj_idx = 0
        
        # 当前对象的临时prompts
        self.current_boxes = []
        self.current_points = []
        
        # 🔥 新增：缓存当前所有对象的合并mask
        self.cached_combined_mask = None
        
        # Box绘制
        self.drawing_box = False
        self.box_start = None
        self.current_rect = None
        
        self.confirmed = False
        self.cancelled = False
        
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.canvas.manager.set_window_title('Multi-Object Mask Editor')
        
        self.update_display()
        
        # 按钮
        ax_done = plt.axes([0.7, 0.02, 0.08, 0.04])
        ax_reset = plt.axes([0.6, 0.02, 0.09, 0.04])
        ax_cancel = plt.axes([0.5, 0.02, 0.09, 0.04])
        ax_new_obj = plt.axes([0.37, 0.02, 0.12, 0.04])
        ax_click_mode = plt.axes([0.25, 0.02, 0.12, 0.04])
        
        self.btn_done = Button(ax_done, 'Done ✓', color='lightgreen')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_cancel = Button(ax_cancel, 'Cancel', color='lightcoral')
        self.btn_new_obj = Button(ax_new_obj, 'New Object', color='lightyellow')
        self.btn_click_mode = Button(ax_click_mode, 'Mode: Box', color='lightblue')
        
        self.btn_done.on_clicked(lambda e: self.on_done())
        self.btn_reset.on_clicked(lambda e: self.on_reset())
        self.btn_cancel.on_clicked(lambda e: self.on_cancel())
        self.btn_new_obj.on_clicked(lambda e: self.new_object())
        self.btn_click_mode.on_clicked(lambda e: self.toggle_click_mode())
        
        # 事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        print("\n🖱️ Multi-Object Mask Editor:")
        print("   Draw boxes/points for current object")
        print("   'New Object' button or N key: Start a new object")
        print("   Each object will be tracked separately")
        print("   U: Undo last box/point")
        print("   R: Reset all")
        print("   Space/Done: Confirm all objects")
        print(f"   Text prompt: '{text_prompt}'")
    
    def toggle_click_mode(self):
        """切换点击/拖拽模式"""
        self.click_mode = "point" if self.click_mode == "box" else "box"
        mode_text = "Point" if self.click_mode == "point" else "Box"
        self.btn_click_mode.label.set_text(f'Mode: {mode_text}')
        self.fig.canvas.draw()
        print(f"   Click mode: {self.click_mode}")
    
    def new_object(self):
        """开始标注新对象"""
        # 保存当前对象
        if len(self.current_boxes) > 0 or len(self.current_points) > 0:
            # 生成当前对象的mask
            self.generate_object_mask()
            
            self.objects.append({
                "boxes": self.current_boxes.copy(),
                "points": self.current_points.copy(),
            })
            
            self.current_boxes = []
            self.current_points = []
            self.current_obj_idx += 1
            
            print(f"\n🆕 Object {self.current_obj_idx} - Start marking new object")
            
            # 🔥 新增：重新生成并显示当前所有对象的mask
            self.regenerate_all_masks()
        else:
            print("   ⚠️ Current object has no prompts")
        
        self.update_display()
    
    def generate_initial_mask(self):
        """用text prompt生成初始mask"""
        try:
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=self.session_id,
                    frame_index=self.frame_idx,
                    object_id=0,
                    text=self.text_prompt,
                )
            )
            
            # 获取并显示mask
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
                
                # 显示initial mask
                print(f"   ✅ Initial mask generated with text")
                self.update_display_with_mask(mask)
            else:
                print(f"   ⚠️ No initial mask generated")
                
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
    
    def on_press(self, event):
        """鼠标按下"""
        if event.inaxes != self.ax:
            return
        
        if self.click_mode == "point":
            # 点模式
            x, y = int(event.xdata), int(event.ydata)
            self.current_points.append([x, y])
            print(f"   ✅ Point at ({x}, {y}) for Object {self.current_obj_idx}")
            self.update_display()
        else:
            # Box模式
            self.drawing_box = True
            self.box_start = (event.xdata, event.ydata)
    
    def on_motion(self, event):
        """鼠标移动"""
        if self.click_mode != "box" or not self.drawing_box:
            return
        
        if event.inaxes != self.ax or self.box_start is None:
            return
        
        if self.current_rect is not None:
            self.current_rect.remove()
        
        x0, y0 = self.box_start
        x1, y1 = event.xdata, event.ydata
        width = x1 - x0
        height = y1 - y0
        
        self.current_rect = Rectangle(
            (x0, y0), width, height,
            fill=False, edgecolor='lime', linewidth=2, linestyle="--",
        )
        self.ax.add_patch(self.current_rect)
        self.fig.canvas.draw_idle()
    
    def on_release(self, event):
        """鼠标释放"""
        if self.click_mode != "box" or not self.drawing_box:
            return
        
        self.drawing_box = False
        
        if event.inaxes != self.ax or self.box_start is None:
            return
        
        if self.current_rect is not None:
            self.current_rect.remove()
            self.current_rect = None
        
        x0, y0 = self.box_start
        x1, y1 = event.xdata, event.ydata
        
        x_min = min(x0, x1)
        x_max = max(x0, x1)
        y_min = min(y0, y1)
        y_max = max(y0, y1)
        
        if abs(x_max - x_min) < 5 or abs(y_max - y_min) < 5:
            return
        
        self.current_boxes.append((x_min, y_min, x_max, y_max))
        print(f"   ✅ Box: ({x_min:.0f}, {y_min:.0f}) -> ({x_max:.0f}, {y_max:.0f}) for Object {self.current_obj_idx}")
        
        self.update_display()
    
    def generate_object_mask(self):
        """为当前对象生成mask"""
        if len(self.current_boxes) == 0 and len(self.current_points) == 0:
            return
        
        boxes_xywh = []
        
        print(f"   🔍 DEBUG: current_boxes={len(self.current_boxes)}, current_points={len(self.current_points)}")
        
        # 添加boxes
        for x_min, y_min, x_max, y_max in self.current_boxes:
            center_x = ((x_min + x_max) / 2) / self.W
            center_y = ((y_min + y_max) / 2) / self.H
            width = (x_max - x_min) / self.W
            height = (y_max - y_min) / self.H
            
            boxes_xywh.append([center_x, center_y, width, height])
            print(f"   🔍 DEBUG: Added box: [{center_x:.3f}, {center_y:.3f}, {width:.3f}, {height:.3f}]")
        
        # 添加points转换的boxes
        for x, y in self.current_points:
            size = 30
            x_min = max(0, x - size/2)
            y_min = max(0, y - size/2)
            x_max = min(self.W, x + size/2)
            y_max = min(self.H, y + size/2)
            
            center_x = ((x_min + x_max) / 2) / self.W
            center_y = ((y_min + y_max) / 2) / self.H
            width = (x_max - x_min) / self.W
            height = (y_max - y_min) / self.H
            
            boxes_xywh.append([center_x, center_y, width, height])
            print(f"   🔍 DEBUG: Added point box: [{center_x:.3f}, {center_y:.3f}, {width:.3f}, {height:.3f}]")
        
        # 🔥 安全检查：确保有boxes
        if len(boxes_xywh) == 0:
            print(f"   ⚠️ No valid boxes for Object {self.current_obj_idx}")
            return
        
        print(f"   🔍 DEBUG: Total boxes_xywh={len(boxes_xywh)}")
        
        try:
            # 为这个对象添加prompt
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=self.session_id,
                    frame_index=self.frame_idx,
                    object_id=self.current_obj_idx,
                    text=self.text_prompt,  # 🔥 修复：添加text prompt
                    boxes_xywh=boxes_xywh,
                    box_labels=[1] * len(boxes_xywh),
                )
            )
            
            # 🔥 新增：获取并显示mask
            outputs = response.get("outputs", {})
            if "out_binary_masks" in outputs and outputs["out_binary_masks"].shape[0] > 0:
                # 获取当前对象的mask
                if len(outputs["out_binary_masks"]) > self.current_obj_idx:
                    mask = outputs["out_binary_masks"][self.current_obj_idx]
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    
                    if mask.shape != (self.H, self.W):
                        mask = cv2.resize(mask.astype(np.uint8), 
                                         (self.W, self.H), 
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                        mask = mask > 0.5
                    
                    # 显示mask
                    print(f"   ✅ Mask generated for Object {self.current_obj_idx}")
                    self.show_current_mask(mask)
                else:
                    print(f"   ✅ Mask generated for Object {self.current_obj_idx}")
            else:
                print(f"   ✅ Request sent for Object {self.current_obj_idx}")
                
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            import traceback
            traceback.print_exc()
    
    def update_display(self):
        """更新显示"""
        self.ax.clear()
        
        # 🔥 使用缓存的mask
        if self.cached_combined_mask is not None:
            overlay = safe_overlay(self.img, self.cached_combined_mask)
            self.ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        else:
            self.ax.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        
        # 显示已完成的对象（用不同颜色）
        colors = ['cyan', 'magenta', 'yellow', 'orange', 'purple']
        for obj_idx, obj in enumerate(self.objects):
            color = colors[obj_idx % len(colors)]
            
            for x_min, y_min, x_max, y_max in obj["boxes"]:
                rect = Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    fill=False, edgecolor=color, linewidth=2, zorder=10
                )
                self.ax.add_patch(rect)
            
            for x, y in obj["points"]:
                circle = Circle((x, y), 6, color=color, fill=True, alpha=0.9, zorder=10)
                self.ax.add_patch(circle)
        
        # 显示当前对象（lime绿色）
        for x_min, y_min, x_max, y_max in self.current_boxes:
            rect = Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                fill=False, edgecolor='lime', linewidth=3, zorder=10
            )
            self.ax.add_patch(rect)
        
        for x, y in self.current_points:
            circle = Circle((x, y), 6, color='lime', fill=True, alpha=0.9, zorder=10)
            self.ax.add_patch(circle)
            self.ax.plot(x, y, '+', color='black', markersize=10, markeredgewidth=2, zorder=11)
        
        # 标题
        current_prompts = len(self.current_boxes) + len(self.current_points)
        mode = "Box" if self.click_mode == "box" else "Point"
        
        title = f'Object {self.current_obj_idx} | {current_prompts} prompt(s) | Total Objects: {len(self.objects)}\nMode: {mode} | Press N for new object'
        
        self.ax.set_title(title, fontsize=12, pad=15)
        self.ax.axis('off')
        
        self.fig.canvas.draw()
    
    def update_display_with_mask(self, mask):
        """更新显示（包含mask overlay）"""
        self.ax.clear()
        
        # 显示带mask的图像
        overlay = safe_overlay(self.img, mask)
        self.ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        
        self.ax.set_title('Initial mask from text prompt\nAdd boxes/points to refine', fontsize=12, pad=15)
        self.ax.axis('off')
        
        self.fig.canvas.draw()
    
    def show_current_mask(self, mask):
        """显示当前对象的mask（带prompts标记）"""
        # 🔥 更新缓存
        self.cached_combined_mask = mask
        
        self.ax.clear()
        
        # 显示带mask的图像
        overlay = safe_overlay(self.img, mask)
        self.ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        
        # 显示已完成的对象prompts（用不同颜色）
        colors = ['cyan', 'magenta', 'yellow', 'orange', 'purple']
        for obj_idx, obj in enumerate(self.objects):
            color = colors[obj_idx % len(colors)]
            
            for x_min, y_min, x_max, y_max in obj["boxes"]:
                rect = Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    fill=False, edgecolor=color, linewidth=2, zorder=10
                )
                self.ax.add_patch(rect)
            
            for x, y in obj["points"]:
                circle = Circle((x, y), 6, color=color, fill=True, alpha=0.9, zorder=10)
                self.ax.add_patch(circle)
        
        # 显示当前对象的prompts（lime绿色）
        for x_min, y_min, x_max, y_max in self.current_boxes:
            rect = Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                fill=False, edgecolor='lime', linewidth=3, zorder=10
            )
            self.ax.add_patch(rect)
        
        for x, y in self.current_points:
            circle = Circle((x, y), 6, color='lime', fill=True, alpha=0.9, zorder=10)
            self.ax.add_patch(circle)
            self.ax.plot(x, y, '+', color='black', markersize=10, markeredgewidth=2, zorder=11)
        
        current_prompts = len(self.current_boxes) + len(self.current_points)
        title = f'Object {self.current_obj_idx} | {current_prompts} prompt(s) | Total Objects: {len(self.objects)}\nMask generated | Press N for new object'
        
        self.ax.set_title(title, fontsize=12, pad=15)
        self.ax.axis('off')
        
        self.fig.canvas.draw()
    
    def regenerate_all_masks(self):
        """重新生成并显示所有对象的合并mask"""
        # 🔥 修复：不需要单独的get_masks API
        # 每次add_prompt的response已经包含所有对象的masks
        # 我们只需要在切换对象时重新请求一次来获取最新的所有masks
        
        try:
            # 使用一个空的text prompt来触发mask生成（获取当前所有对象）
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=self.session_id,
                    frame_index=self.frame_idx,
                    object_id=self.current_obj_idx,  # 当前新对象的ID
                    text=self.text_prompt,
                )
            )
            
            outputs = response.get("outputs", {})
            if "out_binary_masks" in outputs and outputs["out_binary_masks"].shape[0] > 0:
                # 合并所有已存在对象的masks（不包括当前新对象，因为它还没有prompts）
                combined_mask = None
                # 只合并到current_obj_idx-1，因为current_obj_idx是新对象
                for obj_idx in range(min(self.current_obj_idx, len(outputs["out_binary_masks"]))):
                    mask = outputs["out_binary_masks"][obj_idx]
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    
                    if mask.shape != (self.H, self.W):
                        mask = cv2.resize(mask.astype(np.uint8), 
                                         (self.W, self.H), 
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                        mask = mask > 0.5
                    
                    if combined_mask is None:
                        combined_mask = mask
                    else:
                        combined_mask = combined_mask | mask
                
                if combined_mask is not None:
                    # 显示合并后的mask
                    self.show_all_objects_mask(combined_mask)
                    print(f"   ✅ Showing {self.current_obj_idx} existing object(s)")
            
        except Exception as e:
            print(f"   ⚠️ Error regenerating masks: {e}")
            # 如果失败，只显示prompts标记
            self.update_display()
    
    def show_all_objects_mask(self, mask):
        """显示所有对象的合并mask"""
        # 🔥 更新缓存
        self.cached_combined_mask = mask
        
        self.ax.clear()
        
        # 显示带mask的图像
        overlay = safe_overlay(self.img, mask)
        self.ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        
        # 显示所有已完成对象的prompts
        colors = ['cyan', 'magenta', 'yellow', 'orange', 'purple']
        for obj_idx, obj in enumerate(self.objects):
            color = colors[obj_idx % len(colors)]
            
            for x_min, y_min, x_max, y_max in obj["boxes"]:
                rect = Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    fill=False, edgecolor=color, linewidth=2, zorder=10
                )
                self.ax.add_patch(rect)
            
            for x, y in obj["points"]:
                circle = Circle((x, y), 6, color=color, fill=True, alpha=0.9, zorder=10)
                self.ax.add_patch(circle)
        
        title = f'Object {self.current_obj_idx} | Total Objects: {len(self.objects)}\nAll objects shown | Add prompts for new object'
        
        self.ax.set_title(title, fontsize=12, pad=15)
        self.ax.axis('off')
        
        self.fig.canvas.draw()
    
    def on_reset(self):
        """重置"""
        self.objects.clear()
        self.current_boxes.clear()
        self.current_points.clear()
        self.current_obj_idx = 0
        print("   🔄 Reset all")
        self.update_display()
    
    def undo_last(self):
        """撤销最后一个prompt"""
        if len(self.current_boxes) > 0:
            self.current_boxes.pop()
            print("   ↩️ Undo last box")
        elif len(self.current_points) > 0:
            self.current_points.pop()
            print("   ↩️ Undo last point")
        else:
            print("   Nothing to undo in current object")
        
        self.update_display()
    
    def on_key(self, event):
        if event.key == ' ':
            self.on_done()
        elif event.key in ['r', 'R']:
            self.on_reset()
        elif event.key in ['u', 'U']:
            self.undo_last()
        elif event.key in ['n', 'N']:
            self.new_object()
        elif event.key == 'escape':
            self.on_cancel()
        elif event.key in ['p', 'P']:
            self.click_mode = "point"
            self.btn_click_mode.label.set_text('Mode: Point')
            self.fig.canvas.draw()
        elif event.key in ['b', 'B']:
            self.click_mode = "box"
            self.btn_click_mode.label.set_text('Mode: Box')
            self.fig.canvas.draw()
    
    def on_done(self):
        # 保存最后的对象
        if len(self.current_boxes) > 0 or len(self.current_points) > 0:
            self.generate_object_mask()
            self.objects.append({
                "boxes": self.current_boxes.copy(),
                "points": self.current_points.copy(),
            })
        
        if len(self.objects) == 0:
            print("   ⚠️ No objects marked!")
            return
        
        print(f"✅ Confirmed {len(self.objects)} object(s)!")
        self.confirmed = True
        plt.close(self.fig)
    
    def on_cancel(self):
        print("❌ Cancelled")
        self.cancelled = True
        plt.close(self.fig)
    
    def show(self):
        plt.show()
        
        if self.cancelled or len(self.objects) == 0:
            return None
        return len(self.objects)  # 返回对象数量


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--no-interaction", action="store_true",
                        help="无交互模式，只有文本prompt")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=30)
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(exist_ok=True)
    (output_dir / "viz").mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🎬 SAM3 Video Tracking - Multi-Object")
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
    
    # 尝试从drawer masks自动添加 prompts
    num_objects = add_prompts_from_drawer_masks(
        predictor,
        session_id,
        args.start_frame,
        frames[args.start_frame],
        args.prompt,
        video_dir,
    )
    
    if num_objects == 0:
        if args.no_interaction:
            print(f"🎯 No-interaction: add text prompt '{args.prompt}'")
            response = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=args.start_frame,
                    object_id=0,
                    text=args.prompt,
                )
            )
            outputs = response.get("outputs", {})
            if "out_binary_masks" in outputs and outputs["out_binary_masks"].shape[0] > 0:
                num_objects = 1
            else:
                print("❌ No masks from text prompt")
                return
        else:
            # 交互式编辑
            editor = MultiObjectMaskEditor(
                predictor,
                session_id,
                args.start_frame,
                frames[args.start_frame],
                args.prompt
            )
            
            # 生成初始mask
            print(f"🎯 Generating initial mask with text: '{args.prompt}'...")
            editor.generate_initial_mask()
            
            # 显示并标注多个对象
            num_objects = editor.show()
            
            if num_objects is None:
                print("Cancelled")
                return
    else:
        print(f"✅ Auto-added {num_objects} drawer object(s), skip interaction")

    print(f"\n📹 Propagating {num_objects} object(s)...")
    
    # 🔥 简化：一次propagate获取所有对象的masks
    all_masks = {}  # {frame_idx: [mask0, mask1, ...]}
    
    print(f"\n   → Forward propagation")
    for frame_outputs in predictor.propagate_in_video(
        session_id=session_id,
        propagation_direction="forward",
        start_frame_idx=args.start_frame,
        max_frame_num_to_track=len(frames) - args.start_frame,
    ):
        frame_idx, mask_logits = _extract_masks_from_outputs(frame_outputs)
        
        if mask_logits is not None and len(mask_logits) > 0:
            per_obj_masks = []
            for obj_idx in range(min(num_objects, len(mask_logits))):
                mask = mask_logits[obj_idx]
                if isinstance(mask, torch.Tensor):
                    mask = (mask > 0.0).cpu().numpy()
                while mask.ndim > 2:
                    mask = mask[0]
                per_obj_masks.append(mask > 0.5)
            
            if frame_idx is not None and len(per_obj_masks) > 0:
                all_masks[frame_idx] = per_obj_masks
            
    
    if args.start_frame > 0:
        print(f"   → Backward propagation")
        for frame_outputs in predictor.propagate_in_video(
            session_id=session_id,
            propagation_direction="backward",
            start_frame_idx=args.start_frame,
            max_frame_num_to_track=args.start_frame + 1,
        ):
            frame_idx, mask_logits = _extract_masks_from_outputs(frame_outputs)
            
            if mask_logits is not None and len(mask_logits) > 0:
                per_obj_masks = []
                for obj_idx in range(min(num_objects, len(mask_logits))):
                    mask = mask_logits[obj_idx]
                    if isinstance(mask, torch.Tensor):
                        mask = (mask > 0.0).cpu().numpy()
                    while mask.ndim > 2:
                        mask = mask[0]
                    per_obj_masks.append(mask > 0.5)
                
                if frame_idx is not None and len(per_obj_masks) > 0:
                    all_masks[frame_idx] = per_obj_masks
                
    
    print(f"\n✅ Tracked {len(all_masks)} frames\n")
    
    if len(all_masks) == 0:
        print("❌ No masks!")
        return

    # 使用参考mask位置对 SAM 输出进行重排，保证 mask0/1 对应原始位置
    mask_dir = _find_sub_drawers_mask_dir(video_dir)
    reorder_mapping = None
    if mask_dir is not None and args.start_frame in all_masks:
        start_frame_path = frames[args.start_frame]
        start_img = cv2.imread(str(start_frame_path))
        if start_img is not None:
            ref_centroids = _load_reference_centroids(mask_dir, start_img.shape[0], start_img.shape[1])
        else:
            ref_centroids = None
        if ref_centroids:
            start_masks = all_masks[args.start_frame]
            sam_centroids = []
            for mask in start_masks:
                sam_centroids.append(_mask_centroid(mask))
            reorder_mapping = _compute_greedy_mapping(ref_centroids, sam_centroids)
            if reorder_mapping:
                print("🔁 Reorder mapping (ref -> sam):")
                for ref_idx, sam_idx in reorder_mapping.items():
                    ref_name = ref_centroids[ref_idx][0]
                    print(f"   {ref_name} => sam_obj_{sam_idx}")
    
    print("💾 Saving...")
    
    saved_count = 0
    viz_frames = []
    
    per_obj_dirs = {}
    per_obj_viz_dirs = {}
    for obj_idx in range(num_objects):
        obj_dir = output_dir / f"mask{obj_idx}"
        obj_dir.mkdir(exist_ok=True)
        per_obj_dirs[obj_idx] = obj_dir
        viz_dir = output_dir / f"viz_mask{obj_idx}"
        viz_dir.mkdir(exist_ok=True)
        per_obj_viz_dirs[obj_idx] = viz_dir
    
    
    for frame_idx in tqdm(sorted(all_masks.keys())):
        if frame_idx >= len(frames):
            continue
        
        try:
            per_obj_masks = all_masks[frame_idx]
            if reorder_mapping is not None:
                reordered = []
                for desired_idx in range(num_objects):
                    sam_idx = reorder_mapping.get(desired_idx)
                    if sam_idx is None or sam_idx >= len(per_obj_masks):
                        reordered.append(None)
                    else:
                        reordered.append(per_obj_masks[sam_idx])
                per_obj_masks = reordered
            
            img = cv2.imread(str(frames[frame_idx]))
            if img is None:
                continue
            
            combined_mask = None
            for obj_idx, mask in enumerate(per_obj_masks):
                if mask is None:
                    continue
                if mask.shape != (img.shape[0], img.shape[1]):
                    mask = cv2.resize(mask.astype(np.uint8), 
                                     (img.shape[1], img.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
                
                mask_uint8 = (mask * 255).astype(np.uint8)
                cv2.imwrite(str(per_obj_dirs[obj_idx] / f"{frame_idx:04d}.png"), mask_uint8)
                
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = combined_mask | mask
                
                obj_overlay = safe_overlay(img, mask)
                cv2.putText(obj_overlay, f"Frame {frame_idx} | Obj {obj_idx}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imwrite(str(per_obj_viz_dirs[obj_idx] / f"{frame_idx:04d}.png"), obj_overlay)
            
            if combined_mask is None:
                continue
            
            combined_uint8 = (combined_mask * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / "masks" / f"{frame_idx:04d}.png"), combined_uint8)
            
            overlay = safe_overlay(img, combined_mask)
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
        "num_objects": num_objects,
        "num_frames_tracked": len(all_masks),
        "num_frames_saved": saved_count,
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Done!")
    print(f"   Objects: {num_objects}")
    print(f"   Tracked: {len(all_masks)} frames")
    print(f"   Saved: {saved_count} frames")
    print(f"   Video: {output_dir / 'tracking_video.mp4'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

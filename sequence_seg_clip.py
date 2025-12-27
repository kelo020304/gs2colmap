#!/usr/bin/env python3
"""
Optimized Video Sequence Segmentation with SAM3
- Uses image API for each frame (no video API needed)
- Saves binary masks in one folder (one file per frame)
- Saves colored visualizations in another folder (one file per frame)
- Real-time visualization display
- CLIP-based semantic filtering to select best mask
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Optional, Dict
from tqdm import tqdm
import json
import matplotlib
matplotlib.use('TkAgg')  # For real-time display
import matplotlib.pyplot as plt

from demo import SAM3Segmenter, SegmentationResult

# CLIP imports
from transformers import CLIPProcessor, CLIPModel


class VideoSequenceSegmenter:
    """
    Segment video sequences frame by frame
    - Each frame: segment with text prompt
    - Optional: Use CLIP to filter and keep only the most semantically matching mask
    - Save binary masks: masks/frame_0001.png, masks/frame_0002.png, ...
    - Save colored viz: visualizations/frame_0001.png, ...
    - Real-time display while processing
    """
    
    def __init__(self, device: Optional[str] = None, use_clip_filter: bool = False):
        """Initialize segmenter"""
        self.segmenter = SAM3Segmenter(device)
        self.use_clip_filter = use_clip_filter
        
        # Initialize CLIP if needed
        if use_clip_filter:
            print("🔍 Loading CLIP model for semantic filtering...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            self.clip_model = self.clip_model.to(self.clip_device)
            print("✅ CLIP ready!")
        
        print("✅ Video Segmenter ready!")
    
    def _filter_best_mask_by_clip(
        self,
        image: Image.Image,
        result: SegmentationResult,
        target_text: str
    ) -> SegmentationResult:
        """
        Use CLIP to select the mask that best matches the target text
        
        Args:
            image: Original PIL Image
            result: SegmentationResult with multiple masks
            target_text: Target semantic description
            
        Returns:
            SegmentationResult with only the best mask
        """
        if len(result) == 0:
            return result
        
        if len(result) == 1:
            return result  # Only one mask, no need to filter
        
        # Convert image to numpy
        img_array = np.array(image)
        
        best_score = -1
        best_idx = 0
        
        # Evaluate each mask
        for i, mask in enumerate(result.masks):
            mask_2d = np.squeeze(mask)
            
            # Extract masked region (crop to bounding box for efficiency)
            y_indices, x_indices = np.where(mask_2d)
            if len(y_indices) == 0:
                continue
            
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            # Crop and apply mask
            cropped_img = img_array[y_min:y_max+1, x_min:x_max+1]
            cropped_mask = mask_2d[y_min:y_max+1, x_min:x_max+1]
            
            # Create masked image (background = black)
            masked_img = cropped_img.copy()
            masked_img[~cropped_mask] = 1
            
            # CLIP similarity
            pil_masked = Image.fromarray(masked_img)
            
            inputs = self.clip_processor(
                text=[target_text],
                images=pil_masked,
                return_tensors="pt",
                padding=True
            ).to(self.clip_device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                similarity = outputs.logits_per_image[0, 0].item()
            
            if similarity > best_score:
                best_score = similarity
                best_idx = i
        
        # Return only the best mask
        return SegmentationResult(
            masks=result.masks[best_idx:best_idx+1],
            boxes=result.boxes[best_idx:best_idx+1],
            scores=result.scores[best_idx:best_idx+1],
            prompt=f"{result.prompt}_clip_filtered"
        )
    
    def segment_video_sequence(
        self,
        image_dir: Union[str, Path],
        prompt: str,
        pattern: str = "*.png",
        output_dir: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.5,
        top_k: Optional[int] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        realtime_display: bool = True,
        save_binary_masks: bool = True,
        save_colored_viz: bool = True,
        save_masked_objects: bool = True,
        vis_alpha: float = 0.6,
        clip_target_text: Optional[str] = None
    ) -> Dict:
        """
        Segment video sequence
        
        Args:
            image_dir: Directory containing sequential images
            prompt: Text prompt for SAM3 segmentation (e.g., "objects")
            pattern: File pattern (e.g., "*.png", "frame_*.jpg")
            output_dir: Output directory
            confidence_threshold: Minimum confidence score
            top_k: Number of top objects to keep (before CLIP filtering)
            start_frame: Start frame index
            end_frame: End frame index (None = all)
            realtime_display: Show real-time visualization
            save_binary_masks: Save binary masks
            save_colored_viz: Save colored visualizations
            vis_alpha: Visualization transparency
            clip_target_text: If provided, use CLIP to filter masks by this semantic description
            
        Returns:
            Dictionary with results
        """
        image_dir = Path(image_dir)
        image_files = sorted(image_dir.glob(pattern))
        
        if not image_files:
            raise ValueError(f"No images found in {image_dir} with pattern {pattern}")
        
        # Apply frame range
        if end_frame is not None:
            image_files = image_files[start_frame:end_frame]
        else:
            image_files = image_files[start_frame:]
        
        print(f"\n{'='*60}")
        print(f"🎬 SAM3 Video Sequence Segmentation")
        print(f"{'='*60}")
        print(f"📁 Input: {image_dir}")
        print(f"🎯 SAM3 Prompt: '{prompt}'")
        if clip_target_text:
            print(f"🔍 CLIP Filter: '{clip_target_text}'")
        print(f"🖼️  Frames: {len(image_files)}")
        print(f"   First: {image_files[0].name}")
        print(f"   Last: {image_files[-1].name}")
        print(f"{'='*60}\n")
        
        # Setup output directories
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if save_binary_masks:
                binary_dir = output_dir / "binary_masks"
                binary_dir.mkdir(exist_ok=True)
            
            if save_colored_viz:
                viz_dir = output_dir / "colored_visualizations"
                viz_dir.mkdir(exist_ok=True)
            
            if save_masked_objects:
                masked_dir = output_dir / "masked_objects"
                masked_dir.mkdir(exist_ok=True)
        
        # Setup real-time display
        if realtime_display:
            plt.ion()  # Interactive mode
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            fig.canvas.manager.set_window_title('SAM3 Real-time Segmentation')
        
        # Process frames
        all_results = []
        frame_stats = []
        
        for frame_idx, img_path in enumerate(tqdm(image_files, desc="Segmenting frames")):
            # Load image
            image = Image.open(img_path)
            
            # Segment frame
            result = self.segmenter.segment_by_text(
                image,
                prompt=prompt,
                confidence_threshold=confidence_threshold,
                top_k=top_k
            )
            
            # Apply CLIP filtering if requested
            if clip_target_text and self.use_clip_filter and len(result) > 0:
                result = self._filter_best_mask_by_clip(image, result, clip_target_text)
            
            all_results.append(result)
            
            # Get frame name (without extension)
            frame_name = img_path.stem
            
            # Save binary masks (combined into one image)
            if output_dir and save_binary_masks and len(result) > 0:
                # Combine all masks into one binary image
                combined_mask = result.get_combined_mask(threshold=0.0)
                binary_img = Image.fromarray((combined_mask * 255).astype(np.uint8))
                binary_img.save(binary_dir / f"{frame_name}.png")
            
            # Create and save colored visualization
            if output_dir and save_colored_viz and len(result) > 0:
                colored_img = self._create_colored_visualization(
                    img_path,
                    result,
                    alpha=vis_alpha
                )
                colored_img.save(viz_dir / f"{frame_name}.png")
            
            if output_dir and save_masked_objects and len(result) > 0:
                masked_object_img = self._create_masked_object_image(
                    image,
                    result
                )
                masked_object_img.save(masked_dir / f"{frame_name}.png")

            # Real-time display
            if realtime_display and len(result) > 0:
                self._update_realtime_display(ax, img_path, result, vis_alpha)
                plt.pause(0.01)  # Brief pause to update display
            
            # Collect stats
            frame_stats.append({
                "frame": frame_name,
                "num_objects": len(result),
                "scores": result.scores.tolist() if len(result) > 0 else []
            })
        
        # Close real-time display
        if realtime_display:
            plt.ioff()
            plt.close(fig)
        
        # Save metadata
        metadata = {
            "prompt": prompt,
            "clip_filter": clip_target_text,
            "num_frames": len(image_files),
            "total_objects_found": sum(len(r) for r in all_results),
            "avg_objects_per_frame": np.mean([len(r) for r in all_results]),
            "confidence_threshold": confidence_threshold,
            "top_k": top_k,
            "frame_stats": frame_stats
        }
        
        if output_dir:
            with open(output_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"✅ Segmentation Complete!")
        print(f"{'='*60}")
        print(f"📊 Summary:")
        print(f"   Total frames: {len(all_results)}")
        print(f"   Avg objects per frame: {metadata['avg_objects_per_frame']:.1f}")
        print(f"   Total objects found: {metadata['total_objects_found']}")
        if output_dir:
            print(f"\n📂 Output:")
            if save_binary_masks:
                print(f"   Binary masks: {binary_dir}")
            if save_colored_viz:
                print(f"   Colored viz:  {viz_dir}")
            print(f"   Metadata:     {output_dir / 'metadata.json'}")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "results": all_results,
            "metadata": metadata
        }
    
    def _create_masked_object_image(
        self,
        image: Image.Image,
        result: SegmentationResult,
        background_color: int = 255  # White background
        ) -> Image.Image:
        """
        Create image with only masked objects in color, rest is white background
        
        Args:
            image: Original PIL Image
            result: SegmentationResult with masks
            background_color: Background color (0=black, 255=white)
            
        Returns:
            PIL Image with only objects visible
        """
        # Convert to numpy
        img_array = np.array(image)
        
        # Create white background image
        masked_img = np.ones_like(img_array) * background_color
        
        # Apply all masks (keep object pixels from original image)
        combined_mask = result.get_combined_mask(threshold=0.0)
        
        # Copy object pixels
        masked_img[combined_mask > 0] = img_array[combined_mask > 0]
        
        return Image.fromarray(masked_img)
    
    def _create_colored_visualization(
        self,
        image_path: Path,
        result: SegmentationResult,
        alpha: float = 0.6
    ) -> Image.Image:
        """Create colored mask overlay"""
        # Load image
        image = Image.open(image_path)
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Generate distinct colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(result.masks)))
        
        # Create overlay
        overlay = img_array.copy()
        
        for i, mask in enumerate(result.masks):
            mask_2d = np.squeeze(mask)
            color_rgb = colors[i][:3]
            
            # Blend color
            for c in range(3):
                overlay[:, :, c][mask_2d] = (
                    (1 - alpha) * overlay[:, :, c][mask_2d] + 
                    alpha * color_rgb[c]
                )
        
        # Convert back to PIL
        result_img = Image.fromarray((overlay * 255).astype(np.uint8))
        return result_img
    
    def _update_realtime_display(
        self,
        ax,
        image_path: Path,
        result: SegmentationResult,
        alpha: float = 0.6
    ):
        """Update real-time display"""
        ax.clear()
        
        # Load and display image
        image = Image.open(image_path)
        ax.imshow(image)
        
        # Generate colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(result.masks)))
        
        # Overlay masks
        for i, (mask, score) in enumerate(zip(result.masks, result.scores)):
            mask_2d = np.squeeze(mask)
            
            # Create colored overlay
            colored_mask = np.zeros((*mask_2d.shape, 4))
            colored_mask[mask_2d] = [*colors[i][:3], alpha]
            ax.imshow(colored_mask)
        
        ax.axis('off')
        ax.set_title(
            f"{image_path.name} | {len(result)} object(s)",
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SAM3 Video Sequence Segmentation with CLIP Filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage: segment "objects"
  python sequence_seg.py /path/to/frames --prompt "objects" --output ./results
  
  # With CLIP filtering: segment "objects" then keep only "drawer"
  python sequence_seg.py /path/to/frames --prompt "objects" --clip-filter "drawer" -o ./results
  
  # No real-time display (faster)
  python sequence_seg.py /path/to/frames --prompt "objects" --clip-filter "door" -o ./results --no-display
        """
    )
    
    parser.add_argument("image_dir", help="Directory containing sequential images")
    parser.add_argument("--prompt", required=True, help="Text prompt for SAM3 segmentation")
    parser.add_argument("--clip-filter", default=None, help="Use CLIP to filter masks by this semantic description")
    parser.add_argument("--pattern", default="*.png", help="Image file pattern")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--top-k", type=int, default=None, help="Keep top-k objects per frame (before CLIP)")
    
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=None, help="End frame index")
    
    parser.add_argument("--no-display", action="store_true", help="Disable real-time display")
    parser.add_argument("--no-binary", action="store_true", help="Don't save binary masks")
    parser.add_argument("--no-colored", action="store_true", help="Don't save colored visualizations")
    parser.add_argument("--vis-alpha", type=float, default=0.6, help="Visualization transparency")
    
    args = parser.parse_args()
    
    # Initialize with CLIP if needed
    use_clip = args.clip_filter is not None
    segmenter = VideoSequenceSegmenter(use_clip_filter=use_clip)
    
    # Run segmentation
    results = segmenter.segment_video_sequence(
        image_dir=args.image_dir,
        prompt=args.prompt,
        pattern=args.pattern,
        output_dir=args.output,
        confidence_threshold=args.threshold,
        top_k=args.top_k,
        start_frame=args.start,
        end_frame=args.end,
        realtime_display=not args.no_display,
        save_binary_masks=not args.no_binary,
        save_colored_viz=not args.no_colored,
        vis_alpha=args.vis_alpha,
        clip_target_text=args.clip_filter
    )
    
    print("🎉 Done!")


if __name__ == "__main__":
    main()
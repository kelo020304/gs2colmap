#!/usr/bin/env python3
"""
Gaussian Splatting 属性恢复工具
用于从分割后的PLY恢复完整的3DGS属性
"""

import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


class GaussianAttributeRestorer:
    """
    Gaussian Splatting属性恢复器
    
    功能：
    1. 从原始GS模型加载完整属性
    2. 通过最近邻匹配恢复分割后点云的属性
    3. 支持批量处理多个PLY文件
    """
    
    def __init__(self, source_ply_path, verbose=True):
        """
        初始化属性恢复器
        
        Args:
            source_ply_path: 原始完整的GS PLY文件路径
            verbose: 是否打印详细信息
        """
        self.source_ply_path = Path(source_ply_path)
        self.verbose = verbose
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"初始化 Gaussian 属性恢复器")
            print(f"{'='*70}")
            print(f"源文件: {self.source_ply_path}")
        
        # 加载源PLY
        self._load_source_ply()
        
        # 构建KD-Tree用于快速最近邻查询
        self._build_kdtree()
        
        if self.verbose:
            print(f"✓ 初始化完成")
    
    def _load_source_ply(self):
        """加载源PLY文件的所有属性"""
        if not self.source_ply_path.exists():
            raise FileNotFoundError(f"源PLY文件不存在: {self.source_ply_path}")
        
        plydata = PlyData.read(str(self.source_ply_path))
        self.source_vertices = plydata['vertex'].data  # 修复：添加 .data
        
        # 提取位置 (x, y, z)
        self.source_positions = np.stack([
            self.source_vertices['x'],
            self.source_vertices['y'],
            self.source_vertices['z']
        ], axis=1).astype(np.float32)
        
        self.num_source_points = len(self.source_positions)
        
        if self.verbose:
            print(f"源点云数量: {self.num_source_points:,}")
            # 修复：获取dtype的字段名
            field_names = list(self.source_vertices.dtype.names)
            print(f"属性字段数: {len(field_names)}")
            if len(field_names) <= 10:
                print(f"属性字段: {field_names}")
            else:
                print(f"属性字段: {field_names[:5]} ... {field_names[-3:]}")
    
    def _build_kdtree(self):
        """构建KD-Tree用于最近邻查询"""
        if self.verbose:
            print(f"构建KD-Tree...")
        
        self.kdtree = cKDTree(self.source_positions)
        
        if self.verbose:
            print(f"✓ KD-Tree构建完成")
    
    def restore_attributes(self, target_ply_path, output_ply_path=None, 
                          max_distance=0.001, overwrite=False):
        """
        恢复目标PLY文件的Gaussian属性
        
        Args:
            target_ply_path: 待恢复属性的PLY文件（只有x,y,z）
            output_ply_path: 输出文件路径，None则自动生成
            max_distance: 最大匹配距离（米），超过此距离认为无法匹配
            overwrite: 是否覆盖已存在的输出文件
        
        Returns:
            output_ply_path: 输出文件路径
        """
        target_ply_path = Path(target_ply_path)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"恢复 Gaussian 属性")
            print(f"{'='*70}")
            print(f"目标文件: {target_ply_path}")
        
        # 确定输出路径
        if output_ply_path is None:
            output_ply_path = target_ply_path.parent / f"{target_ply_path.stem}_restored.ply"
        else:
            output_ply_path = Path(output_ply_path)
        
        # 检查输出文件是否已存在
        if output_ply_path.exists() and not overwrite:
            print(f"⚠️  输出文件已存在: {output_ply_path}")
            print(f"   使用 overwrite=True 强制覆盖")
            return output_ply_path
        
        # 加载目标PLY
        target_plydata = PlyData.read(str(target_ply_path))
        target_vertices = target_plydata['vertex'].data  # 修复：添加 .data
        
        # 提取目标位置
        target_positions = np.stack([
            target_vertices['x'],
            target_vertices['y'],
            target_vertices['z']
        ], axis=1).astype(np.float32)
        
        num_target_points = len(target_positions)
        
        if self.verbose:
            print(f"目标点云数量: {num_target_points:,}")
        
        # 使用KD-Tree查找最近邻
        if self.verbose:
            print(f"查找最近邻...")
        
        distances, indices = self.kdtree.query(target_positions, k=1)
        
        # 统计匹配情况
        valid_matches = distances < max_distance
        num_valid = valid_matches.sum()
        num_invalid = (~valid_matches).sum()
        
        if self.verbose:
            print(f"匹配统计:")
            print(f"  有效匹配: {num_valid:,} / {num_target_points:,} "
                  f"({num_valid/num_target_points*100:.1f}%)")
            if num_invalid > 0:
                print(f"  ⚠️  无效匹配: {num_invalid:,} "
                      f"(距离 > {max_distance*1000:.1f}mm)")
                print(f"      最大距离: {distances.max()*1000:.2f}mm")
        
        # 恢复属性
        if self.verbose:
            print(f"恢复属性...")
        
        # 复制源顶点属性到目标
        restored_vertices = self.source_vertices[indices]
        
        # 创建新的PLY
        restored_ply = PlyData([
            PlyElement.describe(restored_vertices, 'vertex')
        ], text=False)
        
        # 保存
        output_ply_path.parent.mkdir(parents=True, exist_ok=True)
        restored_ply.write(str(output_ply_path))
        
        if self.verbose:
            print(f"\n✓ 已保存恢复属性的PLY: {output_ply_path}")
            print(f"  点数: {num_target_points:,}")
        
        return output_ply_path
    
    def batch_restore(self, target_ply_paths, output_dir=None, 
                     suffix="_restored", **kwargs):
        """
        批量恢复多个PLY文件的属性
        
        Args:
            target_ply_paths: PLY文件路径列表
            output_dir: 输出目录，None则在原目录
            suffix: 输出文件后缀
            **kwargs: 传递给restore_attributes的其他参数
        
        Returns:
            output_paths: 输出文件路径列表
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"批量恢复 Gaussian 属性")
            print(f"{'='*70}")
            print(f"文件数量: {len(target_ply_paths)}")
        
        output_paths = []
        
        for i, target_path in enumerate(target_ply_paths, 1):
            target_path = Path(target_path)
            
            if self.verbose:
                print(f"\n[{i}/{len(target_ply_paths)}] 处理: {target_path.name}")
            
            # 确定输出路径
            if output_dir is not None:
                output_path = Path(output_dir) / f"{target_path.stem}{suffix}.ply"
            else:
                output_path = target_path.parent / f"{target_path.stem}{suffix}.ply"
            
            # 恢复属性
            result_path = self.restore_attributes(
                target_path, 
                output_path, 
                **kwargs
            )
            
            output_paths.append(result_path)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"批量处理完成")
            print(f"{'='*70}")
            print(f"成功处理: {len(output_paths)} 个文件")
        
        return output_paths


def restore_single_file(source_ply, target_ply, output_ply=None, **kwargs):
    """
    便捷函数：恢复单个文件的属性
    
    Args:
        source_ply: 原始完整GS PLY文件
        target_ply: 待恢复的PLY文件
        output_ply: 输出文件路径
        **kwargs: 其他参数
    
    Returns:
        output_ply_path: 输出文件路径
    """
    restorer = GaussianAttributeRestorer(source_ply)
    return restorer.restore_attributes(target_ply, output_ply, **kwargs)


def restore_batch_files(source_ply, target_ply_list, **kwargs):
    """
    便捷函数：批量恢复多个文件的属性
    
    Args:
        source_ply: 原始完整GS PLY文件
        target_ply_list: 待恢复的PLY文件列表
        **kwargs: 其他参数
    
    Returns:
        output_paths: 输出文件路径列表
    """
    restorer = GaussianAttributeRestorer(source_ply)
    return restorer.batch_restore(target_ply_list, **kwargs)


def main():
    """命令行工具"""
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="恢复Gaussian Splatting属性")
    parser.add_argument("--source", type=str, required=True,
                       help="原始完整的GS PLY文件")
    parser.add_argument("--target", type=str, nargs="+", required=True,
                       help="待恢复的PLY文件（可以多个）")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录（默认：与target相同）")
    parser.add_argument("--suffix", type=str, default="_restored",
                       help="输出文件后缀（默认：_restored）")
    parser.add_argument("--max-distance", type=float, default=0.001,
                       help="最大匹配距离（米，默认：1mm）")
    parser.add_argument("--overwrite", action="store_true",
                       help="覆盖已存在的输出文件")
    parser.add_argument("--quiet", action="store_true",
                       help="静默模式")
    
    args = parser.parse_args()
    
    # 创建恢复器
    restorer = GaussianAttributeRestorer(
        args.source, 
        verbose=not args.quiet
    )
    
    # 批量恢复
    output_paths = restorer.batch_restore(
        args.target,
        output_dir=args.output_dir,
        suffix=args.suffix,
        max_distance=args.max_distance,
        overwrite=args.overwrite
    )
    
    if not args.quiet:
        print(f"\n✓ 全部完成！")
        print(f"输出文件:")
        for path in output_paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Low-Poly Image to Mesh Converter - Version 2
==============================================
Simplified approach using Delaunay triangulation directly.
"""

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import Delaunay
import trimesh
from typing import Tuple


class LowPolyMeshConverter:
    """Convert low-poly raster images to 3D meshes using Delaunay triangulation."""
    
    def __init__(self, 
                 n_colors: int = 32,
                 edge_sample_rate: int = 5,
                 max_points: int = 10000):
        """
        Initialize converter with parameters.
        
        Args:
            n_colors: Number of colors for quantization
            edge_sample_rate: Sample every Nth pixel from edges
            max_points: Maximum number of points for triangulation
        """
        self.n_colors = n_colors
        self.edge_sample_rate = edge_sample_rate
        self.max_points = max_points
        
    def process_image(self, image_path: str, output_path: str) -> trimesh.Trimesh:
        """
        Main pipeline: convert image to mesh.
        
        Args:
            image_path: Path to input low-poly image
            output_path: Path to save output mesh (OBJ format)
            
        Returns:
            trimesh.Trimesh object
        """
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        print("Phase A: Segmentation...")
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        quantized, color_map = self._quantize_colors(denoised)
        
        print("Phase B: Extract sample points...")
        points = self._extract_sample_points(quantized, color_map)
        print(f"  Sampled {len(points)} points")
        
        print("Phase C: Delaunay triangulation...")
        tri = Delaunay(points)
        
        print("Phase D: Color assignment...")
        vertices_3d, faces, colors = self._create_mesh(points, tri, quantized, h, w)
        
        # Create trimesh
        mesh = trimesh.Trimesh(
            vertices=vertices_3d,
            faces=faces,
            vertex_colors=colors,
            process=False
        )
        
        print(f"Saving mesh to: {output_path}")
        mesh.export(output_path)
        
        print(f"✓ Mesh created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh
    
    def _quantize_colors(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize colors using KMeans."""
        h, w, c = image.shape
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        kmeans = MiniBatchKMeans(n_clusters=self.n_colors, 
                                  random_state=42,
                                  batch_size=1000,
                                  n_init=3)
        labels = kmeans.fit_predict(pixels)
        
        centers = kmeans.cluster_centers_.astype(np.uint8)
        quantized_pixels = centers[labels]
        quantized_image = quantized_pixels.reshape(h, w, c)
        
        return quantized_image, centers
    
    def _extract_sample_points(self, quantized: np.ndarray, color_map: np.ndarray) -> np.ndarray:
        """
        Extract sample points from color boundaries and corners.
        """
        h, w = quantized.shape[:2]
        
        # Convert to color indices
        color_indices = np.zeros((h, w), dtype=np.int32)
        for idx, color in enumerate(color_map):
            mask = np.all(quantized == color, axis=2)
            color_indices[mask] = idx
        
        # Detect edges
        edges = np.zeros((h, w), dtype=np.uint8)
        diff_h = np.diff(color_indices, axis=1)
        edges[:, :-1] |= (diff_h != 0).astype(np.uint8) * 255
        diff_v = np.diff(color_indices, axis=0)
        edges[:-1, :] |= (diff_v != 0).astype(np.uint8) * 255
        
        # Get edge pixels
        edge_coords = np.column_stack(np.where(edges > 0))
        
        # Sample points
        if len(edge_coords) > self.max_points:
            indices = np.random.choice(len(edge_coords), self.max_points, replace=False)
            edge_coords = edge_coords[indices]
        elif len(edge_coords) > self.max_points // 2:
            # Sample every Nth point
            indices = np.arange(0, len(edge_coords), self.edge_sample_rate)
            edge_coords = edge_coords[indices]
        
        # Add image corners
        corners = np.array([[0, 0], [0, w-1], [h-1, 0], [h-1, w-1]])
        
        # Combine (note: edge_coords are (y, x), swap to (x, y))
        all_points = np.vstack([
            edge_coords[:, [1, 0]],  # Swap to x, y
            corners[:, [1, 0]]
        ])
        
        return all_points.astype(np.float64)
    
    def _create_mesh(self, points: np.ndarray, tri: Delaunay, 
                     quantized: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create 3D mesh from Delaunay triangulation.
        """
        # Convert 2D points to 3D (center and flip Y)
        vertices_3d = np.zeros((len(points), 3), dtype=np.float64)
        vertices_3d[:, 0] = points[:, 0] - w / 2
        vertices_3d[:, 1] = (h - points[:, 1]) - h / 2
        vertices_3d[:, 2] = 0.0
        
        # Sample colors at each vertex
        colors = np.zeros((len(points), 4), dtype=np.uint8)
        for i, (x, y) in enumerate(points):
            xi = int(np.clip(x, 0, w - 1))
            yi = int(np.clip(y, 0, h - 1))
            colors[i, :3] = quantized[yi, xi]
            colors[i, 3] = 255
        
        return vertices_3d, tri.simplices, colors


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert low-poly raster images to 3D meshes'
    )
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output mesh path (OBJ format)')
    parser.add_argument('--colors', type=int, default=32,
                       help='Number of colors for quantization (default: 32)')
    parser.add_argument('--sample-rate', type=int, default=5,
                       help='Edge sample rate (default: 5)')
    parser.add_argument('--max-points', type=int, default=10000,
                       help='Max points for triangulation (default: 10000)')
    
    args = parser.parse_args()
    
    converter = LowPolyMeshConverter(
        n_colors=args.colors,
        edge_sample_rate=args.sample_rate,
        max_points=args.max_points
    )
    
    converter.process_image(args.input, args.output)


if __name__ == '__main__':
    main()

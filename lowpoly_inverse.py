#!/usr/bin/env python3
"""
Low-Poly Image to Mesh Converter
=================================
Reverse-engineers raster low-poly images into valid 2D/3D triangular meshes.

Pipeline:
    Phase A: Segmentation (denoise + color quantization)
    Phase B: Polygon extraction (contours + simplification)
    Phase C: Constrained Delaunay triangulation with color assignment
"""

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import cKDTree
import triangle
import trimesh
from typing import List, Tuple, Dict
from collections import defaultdict


class LowPolyMeshConverter:
    """Convert low-poly raster images to 3D meshes."""
    
    def __init__(self, 
                 n_colors: int = 24,
                 epsilon_factor: float = 0.02,
                 area_constraint: float = 200.0,
                 vertex_merge_threshold: float = 2.0):
        """
        Initialize converter with parameters.
        
        Args:
            n_colors: Number of colors for quantization (KMeans clusters)
            epsilon_factor: Contour simplification factor (fraction of perimeter)
            area_constraint: Max triangle area for triangulation (prevents huge triangles)
            vertex_merge_threshold: Distance threshold for merging duplicate vertices
        """
        self.n_colors = n_colors
        self.epsilon_factor = epsilon_factor
        self.area_constraint = area_constraint
        self.vertex_merge_threshold = vertex_merge_threshold
        
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
        
        print("Phase A: Segmentation...")
        denoised = self._denoise_image(image)
        quantized, color_map = self._quantize_colors(denoised)
        
        print("Phase B: Polygon extraction...")
        pslg_vertices, pslg_segments = self._extract_polygons(quantized, color_map)
        
        print("Phase C: Constrained triangulation...")
        mesh = self._triangulate_and_color(pslg_vertices, pslg_segments, 
                                           quantized, image.shape)
        
        print(f"Saving mesh to: {output_path}")
        mesh.export(output_path)
        
        print(f"✓ Mesh created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter to smooth surfaces while preserving edges.
        
        Args:
            image: Input RGB image
            
        Returns:
            Denoised image
        """
        # Bilateral filter: smooth while keeping edges sharp
        # d=9: diameter of pixel neighborhood
        # sigmaColor=75: filter sigma in color space
        # sigmaSpace=75: filter sigma in coordinate space
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    def _quantize_colors(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize colors using KMeans clustering.
        
        Args:
            image: Input RGB image
            
        Returns:
            Tuple of (quantized_image, color_palette)
        """
        h, w, c = image.shape
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Use MiniBatchKMeans for faster clustering on large images
        kmeans = MiniBatchKMeans(n_clusters=self.n_colors, 
                                  random_state=42,
                                  batch_size=1000,
                                  n_init=3)
        labels = kmeans.fit_predict(pixels)
        
        # Get quantized colors
        centers = kmeans.cluster_centers_.astype(np.uint8)
        quantized_pixels = centers[labels]
        quantized_image = quantized_pixels.reshape(h, w, c)
        
        return quantized_image, centers
    
    def _extract_polygons(self, quantized: np.ndarray, 
                         color_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract edge-based PSLG from quantized image using global edge detection.
        
        Args:
            quantized: Quantized RGB image
            color_map: Color palette from KMeans
            
        Returns:
            Tuple of (vertices, segments) for PSLG
        """
        h, w = quantized.shape[:2]
        
        # Detect color boundaries using edge detection
        # Convert to grayscale using color index
        color_indices = np.zeros((h, w), dtype=np.int32)
        for idx, color in enumerate(color_map):
            mask = np.all(quantized == color, axis=2)
            color_indices[mask] = idx
        
        # Detect edges where color changes
        edges = np.zeros((h, w), dtype=np.uint8)
        
        # Horizontal edges
        diff_h = np.diff(color_indices, axis=1)
        edges[:, :-1] |= (diff_h != 0).astype(np.uint8) * 255
        
        # Vertical edges
        diff_v = np.diff(color_indices, axis=0)
        edges[:-1, :] |= (diff_v != 0).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Thin edges to single pixel width
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Don't add image boundary - let quantization handle it
        
        # Save debug edge map
        cv2.imwrite("debug_edges.png", edges)
        
        # Find contours from edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        all_vertices = []
        all_segments = []
        vertex_offset = 0
        
        for contour in contours:
            if len(contour) < 3:
                continue
           
            # Simplify contour
            perimeter = cv2.arcLength(contour, closed=True)
            epsilon = self.epsilon_factor * perimeter
            simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
            
            if len(simplified) < 3:
                continue
            
            # Extract vertices
            vertices = simplified.squeeze()
            if vertices.ndim == 1:  # Single point
                continue
            
            n_verts = len(vertices)
            
            # Add vertices
            all_vertices.extend(vertices.tolist())
            
            # Create segments
            for i in range(n_verts):
                v1 = vertex_offset + i
                v2 = vertex_offset + (i + 1) % n_verts
                all_segments.append([v1, v2])
            
            vertex_offset += n_verts
        
        if len(all_vertices) == 0:
            raise ValueError("No edges found in image")
        
        # Convert to numpy arrays
        vertices = np.array(all_vertices, dtype=np.float64)
        segments = np.array(all_segments, dtype=np.int32)
        
        # Merge duplicate vertices
        vertices, segments = self._merge_duplicate_vertices(vertices, segments)
        
        return vertices, segments
    
    def _merge_duplicate_vertices(self, vertices: np.ndarray, 
                                  segments: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge duplicate vertices within threshold distance using KDTree.
        
        Args:
            vertices: Nx2 array of vertex positions
            segments: Mx2 array of segment indices
            
        Returns:
            Tuple of (merged_vertices, updated_segments)
        """
        if len(vertices) == 0:
            return vertices, segments
        
        # Build KDTree for fast nearest neighbor queries
        tree = cKDTree(vertices)
        
        # Find all pairs of vertices within threshold
        pairs = tree.query_pairs(r=self.vertex_merge_threshold, output_type='ndarray')
        
        # Union-find structure to group vertices
        parent = np.arange(len(vertices))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Merge all pairs
        for v1, v2 in pairs:
            union(v1, v2)
        
        # Build vertex map
        vertex_groups = {}
        for i in range(len(vertices)):
            root = find(i)
            if root not in vertex_groups:
                vertex_groups[root] = []
            vertex_groups[root].append(i)
        
        # Create new vertices (average position of merged vertices)
        new_vertices = []
        vertex_map = {}
        for group_vertices in vertex_groups.values():
            new_idx = len(new_vertices)
            avg_pos = vertices[group_vertices].mean(axis=0)
            new_vertices.append(avg_pos)
            for old_idx in group_vertices:
                vertex_map[old_idx] = new_idx
        
        # Update segments with new indices and deduplicate
        new_segments_set = set()
        for seg in segments:
            new_seg = tuple(sorted([vertex_map[seg[0]], vertex_map[seg[1]]]))
            if new_seg[0] != new_seg[1]:  # Skip degenerate segments
                new_segments_set.add(new_seg)
        
        new_segments = list(new_segments_set)
        
        return np.array(new_vertices, dtype=np.float64), np.array(new_segments, dtype=np.int32)
    
    def _triangulate_and_color(self, vertices: np.ndarray, segments: np.ndarray,
                               quantized: np.ndarray, 
                               image_shape: Tuple[int, int, int]) -> trimesh.Trimesh:
        """
        Perform constrained Delaunay triangulation and assign colors.
        
        Args:
            vertices: Nx2 array of vertex positions
            segments: Mx2 array of segment indices
            quantized: Quantized RGB image for color sampling
            image_shape: Original image shape (h, w, c)
            
        Returns:
            trimesh.Trimesh object with vertex colors
        """
        # Build PSLG dictionary for triangle library
        pslg = {
            'vertices': vertices,
            'segments': segments
        }
        
        print(f"  PSLG: {len(vertices)} vertices, {len(segments)} segments")
        
        # Triangulate with flags:
        # p: Planar Straight Line Graph
        # Try without q and a flags first to avoid topological issues
        opts = 'p'
        try:
            tri_result = triangle.triangulate(pslg, opts=opts)
        except RuntimeError as e:
            print(f"Triangulation failed: {e}")
            print(f"Saving PSLG to debug_pslg.npz for inspection")
            np.savez('debug_pslg.npz', vertices=vertices, segments=segments)
            raise
        
        # Extract triangulation results
        tri_vertices = tri_result['vertices']
        tri_faces = tri_result['triangles']
        
        # Convert to 3D coordinates (Z=0, flip Y-axis for 3D convention)
        h = image_shape[0]
        vertices_3d = np.zeros((len(tri_vertices), 3), dtype=np.float64)
        vertices_3d[:, 0] = tri_vertices[:, 0] - image_shape[1] / 2  # Center X
        vertices_3d[:, 1] = (h - tri_vertices[:, 1]) - h / 2  # Flip and center Y
        vertices_3d[:, 2] = 0.0  # Flat mesh
        
        # Assign colors to vertices based on face centroids
        vertex_colors = self._assign_vertex_colors(
            tri_vertices, tri_faces, quantized
        )
        
        # Create trimesh
        mesh = trimesh.Trimesh(
            vertices=vertices_3d,
            faces=tri_faces,
            vertex_colors=vertex_colors,
            process=False  # Don't merge vertices or clean
        )
        
        return mesh
    
    def _assign_vertex_colors(self, vertices: np.ndarray, faces: np.ndarray,
                             quantized: np.ndarray) -> np.ndarray:
        """
        Assign colors to vertices based on face centroids.
        
        Args:
            vertices: Nx2 array of 2D vertex positions (image coordinates)
            faces: Mx3 array of triangle face indices
            quantized: Quantized RGB image
            
        Returns:
            Nx4 array of RGBA colors (0-255)
        """
        h, w = quantized.shape[:2]
        vertex_colors = np.zeros((len(vertices), 4), dtype=np.uint8)
        vertex_face_count = np.zeros(len(vertices), dtype=np.int32)
        
        # For each face, sample color at centroid and assign to vertices
        for face in faces:
            # Calculate centroid
            centroid = vertices[face].mean(axis=0)
            
            # Clamp to image bounds
            x = int(np.clip(centroid[0], 0, w - 1))
            y = int(np.clip(centroid[1], 0, h - 1))
            
            # Sample color
            color = quantized[y, x]
            
            # Assign to all vertices of this face
            for v_idx in face:
                vertex_colors[v_idx, :3] += color
                vertex_face_count[v_idx] += 1
        
        # Average colors for vertices shared by multiple faces
        for i in range(len(vertices)):
            if vertex_face_count[i] > 0:
                vertex_colors[i, :3] = vertex_colors[i, :3] // vertex_face_count[i]
            vertex_colors[i, 3] = 255  # Full alpha
        
        return vertex_colors


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert low-poly raster images to 3D meshes'
    )
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output mesh path (OBJ format)')
    parser.add_argument('--colors', type=int, default=48,
                       help='Number of colors for quantization (default: 48)')
    parser.add_argument('--epsilon', type=float, default=0.005,
                       help='Contour simplification factor (default: 0.005)')
    parser.add_argument('--area', type=float, default=100.0,
                       help='Max triangle area constraint (default: 100.0)')
    parser.add_argument('--merge-threshold', type=float, default=1.0,
                       help='Vertex merge threshold (default: 1.0)')
    
    args = parser.parse_args()
    
    converter = LowPolyMeshConverter(
        n_colors=args.colors,
        epsilon_factor=args.epsilon,
        area_constraint=args.area,
        vertex_merge_threshold=args.merge_threshold
    )
    
    converter.process_image(args.input, args.output)


if __name__ == '__main__':
    main()

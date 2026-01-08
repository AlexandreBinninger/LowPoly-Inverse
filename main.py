#!/usr/bin/env python3
"""
Main entry point for the low-poly image to mesh converter.
Processes input.png and generates output.obj
"""

from lowpoly_inverse_v2 import LowPolyMeshConverter


def main():
    """Process the input image and generate mesh."""
    input_file = "input.png"
    output_file = "output.obj"
    
    print("=" * 60)
    print("Low-Poly Image to Mesh Converter")
    print("=" * 60)
    
    # Create converter with default parameters
    converter = LowPolyMeshConverter(
        n_colors=32,               # Number of color clusters
        edge_sample_rate=5,        # Sample every 5th edge pixel
        max_points=10000           # Max points for triangulation
    )
    
    # Process the image
    mesh = converter.process_image(input_file, output_file)
    
    print("=" * 60)
    print(f"Success! Mesh exported to: {output_file}")
    print(f"  • Vertices: {len(mesh.vertices)}")
    print(f"  • Faces: {len(mesh.faces)}")
    print(f"  • Bounding box: {mesh.bounds}")
    print("=" * 60)
    print("\nTo visualize: python visualize.py")


if __name__ == "__main__":
    main()

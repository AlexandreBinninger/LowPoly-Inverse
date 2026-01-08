#!/usr/bin/env python3
"""
Visualize the generated mesh using Polyscope.
"""

import polyscope as ps
import trimesh
import numpy as np


def main():
    """Load and visualize the output mesh."""
    mesh_file = "output.obj"
    
    print(f"Loading mesh: {mesh_file}")
    mesh = trimesh.load(mesh_file, process=False)
    
    print(f"Mesh info:")
    print(f"  • Vertices: {len(mesh.vertices)}")
    print(f"  • Faces: {len(mesh.faces)}")
    print(f"  • Has colors: {mesh.visual.vertex_colors is not None}")
    
    # Initialize Polyscope
    ps.init()
    ps.set_program_name("Low-Poly Mesh Viewer")
    
    # Register the mesh
    ps_mesh = ps.register_surface_mesh(
        "low_poly_mesh",
        mesh.vertices,
        mesh.faces
    )
    
    # Add vertex colors if available
    if mesh.visual.vertex_colors is not None:
        # Convert colors to float [0, 1]
        colors = mesh.visual.vertex_colors[:, :3] / 255.0
        ps_mesh.add_color_quantity("colors", colors, enabled=True)
    
    # Set camera to see the mesh from front
    ps.set_ground_plane_mode("none")
    
    print("\nShowing mesh in Polyscope viewer...")
    print("Press 'q' to quit")
    
    # Show the viewer
    ps.show()


if __name__ == "__main__":
    main()

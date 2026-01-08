# Low-Poly Image to Mesh Converter

A Python pipeline that reverse-engineers raster low-poly images into 3D triangular meshes.

## Quick Start

```bash
# Install dependencies
uv sync

# Convert input.png to output.obj
uv run python main.py

# Visualize the result
uv run python visualize.py
```

## Features

- **Color Quantization**: KMeans clustering to reduce colors and merge similar regions
- **Edge Detection**: Detect boundaries between color regions
- **Delaunay Triangulation**: Automatic mesh generation from sampled edge points
- **Vertex Coloring**: Per-vertex RGB colors sampled from the quantized image
- **OBJ Export**: Standard 3D mesh format compatible with all viewers

## Pipeline

1. **Segmentation**: Bilateral filter + KMeans (32 colors)
2. **Edge Extraction**: Detect color boundaries
3. **Point Sampling**: Sample edge pixels + corners
4. **Triangulation**: scipy Delaunay triangulation
5. **Color Assignment**: Sample colors at vertices
6. **Export**: Save as OBJ with vertex colors

## Files

- `lowpoly_inverse_v2.py` - Core pipeline implementation
- `main.py` - Entry point for processing input.png
- `visualize.py` - Polyscope 3D viewer
- `input.png` - Sample low-poly wolf image
- `output.obj` - Generated mesh output

## Command-Line Usage

```bash
uv run python lowpoly_inverse_v2.py INPUT OUTPUT [OPTIONS]

Options:
  --colors N          Number of colors for quantization (default: 32)
  --sample-rate N     Sample every Nth edge pixel (default: 5)
  --max-points N      Maximum points for triangulation (default: 10000)
```

## Example

```bash
uv run python lowpoly_inverse_v2.py input.png wolf_mesh.obj --colors 24 --max-points 5000
```

## Dependencies

- opencv-python
- numpy
- scikit-learn
- scipy
- trimesh
- polyscope (for visualization)

All managed via `uv` and `pyproject.toml`.

## Output

The generated mesh includes:
- 2D planar mesh (Z=0) with centered coordinates
- Per-vertex RGB colors
- Triangular faces from Delaunay triangulation
- OBJ format compatible with Blender, MeshLab, polyscope, etc.

## License

MIT

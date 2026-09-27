# 3D Representations

> **One-liner:** Images have one natural format (a grid of pixels); 3D has several -- meshes, point clouds, voxels, and implicit fields like SDFs -- and a 3D generation model such as Hunyuan3D moves between them: point cloud in, a latent set in the middle, a field queried on a grid, and a mesh out via marching cubes. Most 3D port bugs live at those conversions, not in the neural network.

## The Intuition

A 512×512 image is 262,144 pixels in a fixed grid, and every image model agrees on that. 3D has no such agreement, because each representation trades memory, precision and ease of use differently:

```
EXPLICIT: store the surface itself          IMPLICIT: store a function of position

  mesh         vertices + triangles           SDF          f(x, y, z) = signed distance
  point cloud  sampled surface points                        to the surface
  voxels       3D grid of filled/empty         occupancy    f(x, y, z) = inside or not
                                                            (probability / logit)
  -> what renderers, games, 3D printers     -> what neural networks produce well:
     and file formats (GLB, OBJ) consume       smooth, any resolution, easy to learn
```

The surface is where the implicit function crosses a threshold (0 for an SDF): the **iso-surface**. Turning that field into a mesh is the job of **marching cubes**.

---

## How It Actually Works

### Meshes

A triangle mesh is two arrays:

```
  vertices  (V, 3) float    xyz positions
  faces     (F, 3) int      indices into vertices, one row per triangle

  + optional: per-vertex normals, UV coordinates (where each vertex lands
    in a 2D texture image), materials (albedo, metallic, roughness)
```

The **order of the three indices** in a face (its *winding*) defines which side is the outside: by the usual convention, counter-clockwise seen from outside. Renderers cull back faces and lighting uses the normal direction, so a mesh with inverted winding renders inside-out or black.

Formats: **GLB/glTF** (binary, materials and textures, Y axis up), **OBJ** (text, widely supported), **PLY** (also stores point clouds). In Python, `trimesh` reads and writes all three.

### Point clouds and voxels

A **point cloud** is just `(N, 3)` positions, often with normals. It's the easiest thing to sample from a mesh and the usual *input* to 3D encoders. Hunyuan3D's shape VAE, for example, encodes points sampled from the surface, with extra samples on sharp edges.

**Voxels** are a 3D occupancy grid. They're simple but scale as N³: 256³ is 16.8M cells, 512³ is 134M. Few modern generators output voxels directly; the grid shows up instead as the set of points where an implicit field is evaluated.

### Implicit fields: SDF and occupancy

A **signed distance function (SDF)** returns, for any point, its distance to the surface, **negative inside and positive outside** by the usual convention. An **occupancy** field returns how likely the point is to be inside -- as a probability or, from a network, a logit, which is positive inside. The two sign conventions are opposite, and that matters as soon as you extract a mesh (see below).

Networks like implicit fields because they're continuous: the same model can be queried at 64³ for a preview or 512³ for detail.

### Marching cubes: from a field to a mesh

Marching cubes evaluates the field on a grid, looks at the 8 corners of each cell, and places triangles where the sign changes, interpolating the exact crossing along each edge. `skimage.measure.marching_cubes` is the standard implementation; it runs on the CPU, on NumPy arrays.

Measured on an analytic sphere (radius 0.5, box ±1.01), M2 Pro CPU:

```
  resolution   grid points     grid (fp32)   marching cubes   triangles
     128          2.1 M            9 MB           23 ms          37,784
     256         17.0 M           68 MB          156 ms         151,544

  Doubling the resolution: 8x the points, ~4x the triangles
```

Marching cubes itself is fast. What's expensive is *filling* the grid when each value comes from a neural network.

### How a latent 3D generator produces a field

Hunyuan3D-2.1 (and the 3DShape2VecSet family it builds on) represents a shape as a **latent set**: 4,096 tokens, with no grid or spatial layout. A diffusion transformer generates that set from an image. To get geometry back, a **geometry decoder** answers "what is the field value at point (x, y, z)?":

```
  query point (x, y, z)
     |  Fourier embedding: sin/cos of the coordinates at several frequencies
     v
  cross-attention: the point's embedding attends to the 4,096 latent tokens
     |
     v
  one scalar: the field value (a logit, positive inside)
```

Filling a 257³ grid means 17 million queries, processed in chunks (`num_chunks = 10000` in Hunyuan3D). Just the attention part of one 10,000-point chunk against 4,096 latents (width 1024, 16 heads, fp16 -- the values in Hunyuan3D-2.1's VAE config) takes 44 ms on an M2 Pro with MLX 0.32.2:

```
  resolution   queries    chunks    attention alone
     128        2.1 M       215         9 s
     256       17.0 M     1,697        74 s
     384       57.1 M     5,707       250 s
```

That's why real pipelines don't decode the full grid. **Hierarchical decoding** (Hunyuan3D's `HierarchicalVolumeDecoding`, and its faster FlashVDM variant) decodes a coarse 65³ grid, keeps only the cells where the sign changes, and refines those at 129³ and then 257³. On the sphere test, the same algorithm queries **0.89M points instead of 17.0M -- 19x fewer**. Cells that are never queried are marked with a sentinel (NaN), which `skimage`'s marching cubes skips.

### Other representations you'll meet

- **Triplanes**: three 2D feature planes (XY, XZ, YZ); a point's feature is the sum of its three projections. This makes a 3D field cheap to store and lets 2D convolutions generate it.
- **3D Gaussian splatting**: millions of small colored, oriented Gaussians rendered by projecting them onto the image. Excellent for view synthesis, but not a watertight surface.
- **NeRF**: a network mapping (position, view direction) to color and density, rendered by marching rays. Mostly superseded by Gaussians for speed.

---

## Why It Matters for MLX

### What runs in MLX, and what doesn't

In a 3D generation pipeline, the neural parts -- the image encoder, the diffusion transformer, the geometry decoder, the texture diffusion -- port to MLX like any other model. The geometry processing usually stays on the CPU in NumPy-based libraries: marching cubes (`scikit-image`, `PyMCubes`), mesh cleanup and remeshing (`trimesh`, `pymeshlab`), UV unwrapping (`xatlas`). Porting those to Metal is rarely worth it: they're not the bottleneck. Rasterization for texture baking is the exception, and the Hunyuan3D MLX port implements its own Metal rasterizer.

The port boundary is therefore an `mx.array → np.ndarray` conversion. Evaluate before converting ([Lazy Evaluation](15-lazy-evaluation.md)), and convert the whole grid once, not chunk by chunk inside a Python loop.

### The sign convention decides which way the mesh faces

`skimage.measure.marching_cubes` orients triangles for a field that is **negative inside**. Feed it an occupancy logit (positive inside) and every face is inverted. Measured on the sphere, using the mesh's signed volume (positive when normals point outward):

```
  field                               signed volume     normals
  SDF, negative inside                  +0.5233          outward  (true volume 0.5236)
  occupancy logit, positive inside      -0.5233          INWARD
```

Hunyuan3D's decoder outputs positive-inside logits, so its reference pipeline reverses every face (`faces[:, ::-1]`) before export, and the MLX port has to do the same. Otherwise texture baking, which rejects back-facing texels, fails.

### Reproduce the reference's geometry, quirks included

The Hunyuan3D reference maps marching-cubes vertex indices to world space as `index / grid_size * bbox_size + bbox_min`, with `grid_size = resolution + 1`. The exact mapping divides by `grid_size - 1`. The difference is small but measurable:

```
  mapping                                 sphere radius   center
  index / (grid_size - 1)   (exact)          0.5000       (0, 0, 0)
  index / grid_size         (reference)      0.4962       (-0.0078, -0.0078, -0.0078)
```

A port that "fixes" this no longer matches the reference mesh. Keep the reference behavior for parity, and document it.

### Port failures

```
3D PORT FAILURES

  Symptom                                    Likely cause
  -------                                    ------------
  Mesh renders black / inside-out,           face winding inverted: occupancy
    texture bake empty                         (positive inside) fed to marching
                                               cubes without reversing faces
  Mesh slightly smaller / shifted vs         vertex index -> world mapping
    reference                                  differs (grid_size vs grid_size - 1)
  Shape mirrored or rotated                  meshgrid indexing "xy" vs "ij", or
                                               axis convention (glTF Y-up vs Z-up)
  Blocky "staircase" surface                 grid resolution too low, or boolean
                                               occupancy instead of continuous logits
                                               (marching cubes needs values to
                                               interpolate)
  Holes where the surface should be          hierarchical decoding skipped cells
                                               (sentinel values left in), or
                                               near-surface mask too tight
  Out of memory while decoding               too large a num_chunks, or the whole
                                               grid's queries built as one array
  Decoding takes minutes                     dense grid decoding at high
                                               resolution: use hierarchical decoding
```

---

## Key Terms

| Term | Definition |
|------|------------|
| **Mesh** | A surface made of vertices `(V, 3)` and triangle faces `(F, 3)` indexing them. |
| **Winding** | The order of a triangle's vertex indices; decides which side is outside (normal direction). |
| **Point cloud** | A set of 3D points `(N, 3)`, often sampled from a surface; a common input to 3D encoders. |
| **Voxel grid** | A 3D grid of cells, each empty or filled (or holding a value); memory grows as N³. |
| **SDF (signed distance function)** | Distance from a point to the surface, negative inside and positive outside by convention. |
| **Occupancy field** | Probability (or logit) that a point is inside the shape; positive-inside, the opposite sign of an SDF. |
| **Iso-surface** | The surface where an implicit field equals a chosen level (0 for SDFs and logits). |
| **Marching cubes** | Algorithm that extracts a triangle mesh from a field sampled on a grid by interpolating sign changes in each cell. |
| **Latent set** | A shape encoded as an unordered set of latent tokens (4,096 in Hunyuan3D-2.1) rather than a grid. |
| **Geometry decoder** | Network that maps a query point and the latent set to a field value, via Fourier embedding and cross-attention. |
| **Hierarchical volume decoding** | Coarse-to-fine grid evaluation that only refines cells near the surface. |
| **Triplane / 3D Gaussian splatting / NeRF** | Alternative 3D representations: three 2D feature planes; many rendered Gaussians; a ray-marched neural field. |

---

## Sources

- Lorensen, W. E., & Cline, H. E. (1987). "Marching Cubes: A High Resolution 3D Surface Construction Algorithm." *SIGGRAPH*.
- Lewiner, T., et al. (2003). "Efficient Implementation of Marching Cubes' Cases with Topological Guarantees" (the `method="lewiner"` variant). `scikit-image` docs: [scikit-image.org/docs/stable/api/skimage.measure.html](https://scikit-image.org/docs/stable/api/skimage.measure.html)
- Park, J. J., et al. (2019). "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation." [arxiv.org/abs/1901.05103](https://arxiv.org/abs/1901.05103)
- Zhang, B., et al. (2023). "3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models." [arxiv.org/abs/2301.11445](https://arxiv.org/abs/2301.11445)
- Hunyuan3D Team (2025). "Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material." [arxiv.org/abs/2506.15442](https://arxiv.org/abs/2506.15442)
- Kerbl, B., et al. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." [arxiv.org/abs/2308.04079](https://arxiv.org/abs/2308.04079)
- Mildenhall, B., et al. (2020). "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." [arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934)
- Khronos glTF 2.0 specification (coordinate system: +Y up): [registry.khronos.org/glTF/specs/2.0/glTF-2.0.html](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
- Hunyuan3D-2.1 MLX port (`hy3dshape/.../surface_extractors.py`, `volume_decoders.py`, `pipeline_mlx.py`): face flip, vertex mapping, hierarchical decoding. [github.com/dgrauet/Hunyuan3D-2.1-mlx](https://github.com/dgrauet/Hunyuan3D-2.1-mlx)
- Measurements on this page: analytic sphere with `scikit-image` marching cubes; attention timing with MLX 0.32.2 on an M2 Pro; hierarchical query count from a re-implementation of the reference algorithm on the sphere.

---

## See Also

- [Latent Space](07-latent-space.md) -- the 2D version of "encode to a compact latent, decode back"
- [Attention](06-attention.md) -- the cross-attention the geometry decoder uses
- [Diffusion](08-diffusion.md) -- how the latent set is generated (flow matching in Hunyuan3D-2.1)
- [Positional Encoding](11-positional-encoding.md) -- sinusoidal encodings, the same idea as the Fourier embedding of query points
- [Lazy Evaluation](15-lazy-evaluation.md) -- evaluating before handing arrays to NumPy geometry code

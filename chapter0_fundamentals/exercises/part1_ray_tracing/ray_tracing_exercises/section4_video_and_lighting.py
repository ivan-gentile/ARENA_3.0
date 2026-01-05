import os
import sys
from pathlib import Path
from typing import Callable
from functools import partial

import torch as t
from torch import Tensor
import einops
from jaxtyping import Float, Bool
from tqdm import tqdm
import plotly.express as px

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from section3_triangles import make_rays_2d

def rotation_matrix(theta: Float[Tensor, ""]) -> Float[Tensor, "rows cols"]:
    """
    Creates a rotation matrix representing a counterclockwise rotation of `theta` around the y-axis.
    """
    return t.tensor(
        [
            [t.cos(theta), 0.0, t.sin(theta)],
            [0.0, 1.0, 0.0],
            [-t.sin(theta), 0.0, t.cos(theta)],
        ]
    )

def raytrace_mesh_video(
    rays: Float[Tensor, "nrays points dim"],
    triangles: Float[Tensor, "ntriangles points dims"],
    rotation_matrix: Callable[[float], Float[Tensor, "rows cols"]],
    raytrace_function: Callable,
    num_frames: int,
) -> Bool[Tensor, "nframes nrays"]:
    """
    Creates a stack of raytracing results, rotating the triangles by `rotation_matrix` each frame.
    """
    result = []
    theta = t.tensor(2 * t.pi) / num_frames
    R = rotation_matrix(theta)
    for theta in tqdm(range(num_frames)):
        triangles = triangles @ R
        result.append(raytrace_function(rays, triangles))
        if t.cuda.is_available():
            t.cuda.empty_cache()
    return t.stack(result, dim=0)

def raytrace_mesh_gpu(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.

    All computations should be performed on the GPU.
    """
    NR = rays.size(0)
    NT = triangles.size(0)
    device = "cuda" if t.cuda.is_available() else "cpu"
    triangles = triangles.to(device)
    rays = rays.to(device)

    # Each triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    triangles_rep = einops.repeat(triangles, "NT pts dims -> pts NR NT dims", NR=NR)
    A, B, C = triangles_rep

    # Each ray is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    rays_rep = einops.repeat(rays, "NR pts dims -> pts NR NT dims", NT=NT)
    O, D = rays_rep

    # Define matrix on left hand side of equation
    mat: Float[Tensor, "NR NT 3 3"] = t.stack([-D, B - A, C - A], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets: Float[Tensor, "NR NT"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3).to(device)

    # Define vector on the right hand side of equation
    vec: Float[Tensor, "NR NT 3"] = O - A

    # Solve eqns (note, s is the distance along ray)
    sol: Float[Tensor, "NR NT 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(-1)
    
    # Scale s by Dx to get the distance along ray
    s *= D[..., 0]

    # Get boolean of intersects, and use it to set distance = infinity when there is no intersection
    intersects = (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular
    s[~intersects] = t.inf

    # Get the minimum distance (over all triangles) for each ray
    return einops.reduce(s, "NR NT -> NR", "min").cpu()

def raytrace_mesh_lambert(
    rays: Float[Tensor, "nrays points=2 dims=3"],
    triangles: Float[Tensor, "ntriangles points=3 dims=3"],
    light: Float[Tensor, "dims=3"],
    ambient_intensity: float,
    device: str = "cuda" if t.cuda.is_available() else "cpu",
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the intensity of light hitting the triangle it intersects with (or zero if
    no intersection).
    """
    NR = rays.size(0)
    NT = triangles.size(0)
    triangles = triangles.to(device)
    rays = rays.to(device)

    # Each triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    triangles_repeated = einops.repeat(triangles, "NT pts dims -> pts NR NT dims", NR=NR)
    A, B, C = triangles_repeated

    # Each ray is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    rays_repeated = einops.repeat(rays, "NR pts dims -> pts NR NT dims", NT=NT)
    O, D = rays_repeated

    # Define matrix on left hand side of equation
    mat: Float[Tensor, "NR NT 3 3"] = t.stack([-D, B - A, C - A], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets: Float[Tensor, "NR NT"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3).to(device)

    # Define vector on the right hand side of equation
    vec: Float[Tensor, "NR NT 3"] = O - A

    # Solve eqns (note, s is the distance along ray)
    sol: Float[Tensor, "NR NT 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(-1)  # each shape [NR, NT]
    
    # Scale s by Dx to get the distance along ray
    s *= D[..., 0]

    # Get boolean of intersects, and use it to set distance = infinity when there is no intersection
    intersects = (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular
    s[~intersects] = float("inf")

    # Get the minimum distance (over all triangles) for each ray
    closest_distances, closest_triangles_for_each_ray = s.min(dim=-1)  # both shape [NR]

    # Get the intensity by taking dot product of triangle normals & light vector
    normals = t.cross(
        triangles[:, 2] - triangles[:, 0], triangles[:, 1] - triangles[:, 0], dim=1
    )  # shape [NT dims]
    normals /= normals.norm(dim=1, keepdim=True)
    intensity_per_triangle = einops.einsum(normals, light.to(device), "nt dims, dims -> nt")
    intensity_per_triangle_signed = t.where(intensity_per_triangle > 0, intensity_per_triangle, 0.0)

    # Get intensity for each ray, and add ambient intensity where there's an intersection
    intensity = intensity_per_triangle_signed[closest_triangles_for_each_ray] + ambient_intensity

    # Set to zero if the ray doesn't intersect with any triangle
    intensity = t.where(closest_distances.isfinite(), intensity, 0.0)

    return intensity.cpu()

def display_video(distances: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly.
    """
    px.imshow(
        distances,
        animation_frame=0,
        origin="lower",
        zmin=0.0,
        zmax=distances[distances.isfinite()].quantile(0.99).item(),
        color_continuous_scale="viridis_r",
    ).update_layout(
        coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video"
    ).show()

def display_video_with_lighting(intensity: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results with lighting.
    """
    px.imshow(
        intensity,
        animation_frame=0,
        origin="lower",
        color_continuous_scale="magma",
    ).update_layout(
        coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video (lighting)"
    ).show()

if __name__ == "__main__":
    # Testing rotation_matrix
    tests.test_rotation_matrix(rotation_matrix)
    print("All tests for section 4 passed!")

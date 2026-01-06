import os
import sys
from pathlib import Path
import torch as t
from torch import Tensor
import einops
from jaxtyping import Float, Bool
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
from part1_ray_tracing.utils import render_lines_with_plotly
from plotly_utils import imshow

Point = Float[Tensor, "points=3"]

def make_rays_2d(
    num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float
) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    n_pixels = num_pixels_y * num_pixels_z
    ygrid = t.linspace(-y_limit, y_limit, num_pixels_y)
    zgrid = t.linspace(-z_limit, z_limit, num_pixels_z)
    rays = t.zeros((n_pixels, 2, 3), dtype=t.float32)
    rays[:, 1, 0] = 1
    rays[:, 1, 1] = einops.repeat(ygrid, "y -> (y z)", z=num_pixels_z)
    rays[:, 1, 2] = einops.repeat(zgrid, "z -> (y z)", y=num_pixels_y)
    return rays

def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    s, u, v = t.linalg.solve(t.stack([-D, B - A, C - A], dim=1), O - A)
    return ((s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)).item()

def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    NR = rays.size(0)

    # Triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    # Each element of `rays` is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    O, D = rays.unbind(dim=1)

    # Define matrix on left hand side of equation
    mat: Float[Tensor, "NR 3 3"] = t.stack([-D, B - A, C - A], dim=-1)

    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets: Float[Tensor, "NR"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # Define vector on the right hand side of equation
    vec = O - A

    # Solve eqns
    sol: Float[Tensor, "NR 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    # Return boolean of (matrix is nonsingular) && (solution is in correct range implying intersection)
    return (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular

def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    NR = rays.size(0)
    NT = triangles.size(0)

    # Each triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    triangles = einops.repeat(triangles, "NT pts dims -> pts NR NT dims", NR=NR)
    A, B, C = triangles

    # Each ray is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    rays = einops.repeat(rays, "NR pts dims -> pts NR NT dims", NT=NT)
    O, D = rays

    # Define matrix on left hand side of equation
    mat: Float[Tensor, "NR NT 3 3"] = t.stack([-D, B - A, C - A], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets: Float[Tensor, "NR NT"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # Define vector on the right hand side of equation
    vec: Float[Tensor, "NR NT 3"] = O - A

    # Solve eqns (note, s is the distance along ray)
    sol: Float[Tensor, "NR NT 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(-1)
 
    # Scale s by Dx to get the distance along ray
    s *= D[..., 0]

    # Get boolean of intersects, and use it to set distance = infinity when there is no intersection
    intersects = (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular
    s[~intersects] = float("inf")

    # Get the minimum distance (over all triangles) for each ray
    return einops.reduce(s, "NR NT -> NR", "min")

if __name__ == "__main__":
    # Testing make_rays_2d
    rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
    print("rays_2d shape:", rays_2d.shape)

    # Testing triangle_ray_intersects
    tests.test_triangle_ray_intersects(triangle_ray_intersects)

    # Testing raytrace_triangle
    A = t.tensor([1, 0.0, -0.5])
    B = t.tensor([1, -0.5, 0.0])
    C = t.tensor([1, 0.5, 0.5])
    test_triangle = t.stack([A, B, C], dim=0)
    num_pixels_y = num_pixels_z = 15
    y_limit = z_limit = 0.5
    rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    intersects = raytrace_triangle(rays2d, test_triangle)
    img = intersects.reshape(num_pixels_y, num_pixels_z).int()
    # imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

    # Testing raytrace_mesh
    triangles = t.load(section_dir / "pikachu.pt", weights_only=True)
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1
    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh(rays, triangles)
    print("Mesh raytracing completed!")
    
    print("All tests for section 3 passed!")

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

Point = Float[Tensor, "points=3"]

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
    
#tests.test_triangle_ray_intersects(triangle_ray_intersects)

#%%

# def raytrace_triangle(
#     rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
#     triangle: Float[Tensor, "trianglePoints=3 dims=3"],
# ) -> Bool[Tensor, "nrays"]:
#     """
#     For each ray, return True if the triangle intersects that ray.
#     """
#     D = rays[:, 1, :]
#     nrays = D.size(0)
#     triangle_arr = einops.repeat(triangle, "a b -> nray a b", nray = nrays)
#     print(D.size(), triangle_arr.size())
#     A = triangle_arr[:,0,:]
#     B = triangle_arr[:,1,:]
#     C = triangle_arr[:,2,:]

#     sol = t.linalg.solve(t.stack([-D, B-A, C-A], dim=-1), -A)
#     s = sol[:,0]
#     u = sol[:,1]
#     v = sol[:,2]

#     return ((s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1))
# A = t.tensor([1, 0.0, -0.5])
# B = t.tensor([1, -0.5, 0.0])
# C = t.tensor([1, 0.5, 0.5])
# num_pixels_y = num_pixels_z = 15
# y_limit = z_limit = 0.5

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
    assert A.shape == (NR, 3)

    # Each element of `rays` is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    O, D = rays.unbind(dim=1)
    assert O.shape == (NR, 3)

    # Define matrix on left hand side of equation
    mat: Float[Tensor, "NR 3 3"] = t.stack([-D, B - A, C - A], dim=-1)

    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    # Note - this works because mat[is_singular] has shape (NR_where_singular, 3, 3), so we
    # can broadcast the identity matrix to that shape.
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

# # Plot triangle & rays
# test_triangle = t.stack([A, B, C], dim=0)
# rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
# triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
# render_lines_with_plotly(rays2d, triangle_lines)

# # Calculate and display intersections
# intersects = raytrace_triangle(rays2d, test_triangle)
# print(intersects)
# img = intersects.reshape(num_pixels_y, num_pixels_z).int()
# imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %%


triangles = t.load(section_dir / "pikachu.pt", weights_only=True)


def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    NR = rays.size(0)
    NT = triangles.size(0)
    print("DEBUG")
    print(NT)

    # Triangles_arr is [ntriangles[[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]] 
    triangles_arr = einops.repeat(triangles, " ntriangles pts dims -> ntriangles pts NR dims", NR=NR)
    rays_arr = einops.repeat(rays, " nrays pts dims ->  NT pts nrays dims", NT=NT).clone()

    A_arr = triangles_arr[:,0,:,:]
    B_arr = triangles_arr[:,1,:,:]
    C_arr = triangles_arr[:,2,:,:]

    O_arr = rays_arr[:,0,:,:]
    D_arr = rays_arr[:,1,:,:]
    mat: Float[Tensor, "NR NT 3 3"] = t.stack([-D_arr, B_arr - A_arr, C_arr - A_arr], dim=-1)
    vec_arr = O_arr - A_arr
    # assert A.shape == (NR, 3)

    # # Each element of `rays` is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    # O, D = rays.unbind(dim=1)
    # assert O.shape == (NR, 3)

    # # Define matrix on left hand side of equation
    # mat: Float[Tensor, "NR 3 3"] = t.stack([-D, B - A, C - A], dim=-1)

    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    # Note - this works because mat[is_singular] has shape (NR_where_singular, 3, 3), so we
    # can broadcast the identity matrix to that shape.
    dets: Float[Tensor, "NR"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # Solve eqns
    sol: Float[Tensor, "NR NT 3"] = t.linalg.solve(mat, vec_arr)
    s, u, v = sol.unbind(dim=-1)
    print(sol.size())


    # # Return boolean of (matrix is nonsingular) && (solution is in correct range implying intersection)

    #distance
    distance = s
    distance[is_singular] = t.inf
    distance[~((s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1))] = t.inf 

    print(distance.shape)
    print(distance)



    return einops.reduce(distance, "NT NR -> NR", "min")


num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-3.5,0.0, -2.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()
# %%

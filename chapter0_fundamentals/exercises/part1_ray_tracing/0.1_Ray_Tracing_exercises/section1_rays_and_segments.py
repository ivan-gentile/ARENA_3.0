import os
import sys
from pathlib import Path
import torch as t
from torch import Tensor
from jaxtyping import Float

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

def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    return rays

def intersect_ray_1d(
    ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]
) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    # Get the x and y coordinates (ignore z)
    ray = ray[:, :2]
    segment = segment[:, :2]

    # Ray is [[Ox, Oy], [Dx, Dy]]
    O, D = ray
    # Segment is [[L1x, L1y], [L2x, L2y]]
    L_1, L_2 = segment

    # Create matrix and vector, and solve equation
    mat = t.stack([D, L_1 - L_2], dim=-1)
    vec = L_1 - O

    # Solve equation (return False if no solution)
    try:
        sol = t.linalg.solve(mat, vec)
    except RuntimeError:
        return False

    # If there is a solution, check the soln is in the correct range for there to be an intersection
    u = sol[0].item()
    v = sol[1].item()
    return (u >= 0.0) and (v >= 0.0) and (v <= 1.0)

if __name__ == "__main__":
    rays1d = make_rays_1d(9, 10.0)
    print("rays1d shape:", rays1d.shape)
    
    # Testing intersect_ray_1d
    tests.test_intersect_ray_1d(intersect_ray_1d)
    tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
    print("All tests for section 1 passed!")

import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import from other sections
sys.path.append(str(Path(__file__).parent))

from section1_rays_and_segments import make_rays_1d, intersect_ray_1d
from section2_batched_operations import intersect_rays_1d
from section3_triangles import make_rays_2d, triangle_ray_intersects, raytrace_triangle, raytrace_mesh
from section4_video_and_lighting import rotation_matrix

import part1_ray_tracing.tests as tests

def run_all_tests():
    print("Running tests for Section 1...")
    tests.test_intersect_ray_1d(intersect_ray_1d)
    tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
    print("Section 1 passed!")

    print("\nRunning tests for Section 2...")
    tests.test_intersect_rays_1d(intersect_rays_1d)
    tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
    print("Section 2 passed!")

    print("\nRunning tests for Section 3...")
    tests.test_triangle_ray_intersects(triangle_ray_intersects)
    print("Section 3 passed!")

    print("\nRunning tests for Section 4...")
    tests.test_rotation_matrix(rotation_matrix)
    print("Section 4 passed!")

    print("\nAll sections passed successfully!")

if __name__ == "__main__":
    run_all_tests()

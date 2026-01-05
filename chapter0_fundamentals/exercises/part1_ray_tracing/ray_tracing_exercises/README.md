# Ray Tracing Exercises

This folder contains the converted exercises from the `0.1_Ray_Tracing_exercises.ipynb` Jupyter notebook. 

## Files

- `section1_rays_and_segments.py`: Introduction to rays and segments, and basic 1D ray-segment intersection.
- `section2_batched_operations.py`: Implementation of batched ray-segment intersection.
- `section3_triangles.py`: Triangle-ray intersection, single-triangle rendering, and mesh rendering.
- `section4_video_and_lighting.py`: Rotation matrices, video rendering, GPU usage, and Lambertian lighting.
- `test_all.py`: A script to run all the tests for the above sections.

## How to run

You can run each file individually to check your progress and see visualizations (which will open in your browser via Plotly).

```bash
python section1_rays_and_segments.py
python section2_batched_operations.py
python section3_triangles.py
python section4_video_and_lighting.py
```

To run all tests at once:

```bash
python test_all.py
```

## Note on Visualizations

Since these are `.py` files, Plotly visualizations will attempt to open in your default web browser. If you are running on a remote server without a browser, you might need to use a different Plotly renderer or save the figures to HTML files.

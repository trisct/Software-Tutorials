# Checking Coordinates

This is a simple program mainly to get you familiar with OpenGL routine.

## Clip-space Coordinates

In this program, the vertex shader does not modify the vertex coordinates, and directly sets `gl_Position` to the data from the input buffer, which means `g_vertex_buffer_data` stores _clip-space_ coordinates directly.

`gl_Position` expects clip-space coordinates in the form of `(x,y,z,w)`. Here, `x`, `y` range in `[-1, 1]` and corresponds to the image coordinates. You can think of `z` as _depth_, which is used to clip triangles. Experiments show that, by default, fragments with depth outside the range `[-1, 1]` will be clipped. `w` is used for homogeneous normalization.
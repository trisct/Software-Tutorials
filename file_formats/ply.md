# PLY Format Explained

`ply` has both a binary format and a text (ascii format). We mainly introduce the latter.

`ply` formats consist of a header and a elements. A comprehensive example is as follows.

```
ply
format ascii 1.0
comment XXX generated
element vertex 4
property float x
property float y
property float z
property float nx
property float ny
property float nz
property int flags
property uchar red
property uchar green
property uchar blue
property uchar alpha
property float quality
element face 2
property list uchar int vertex_indices
property int flags
property uchar red
property uchar green
property uchar blue
property uchar alpha
property float quality
end_header
0 0 0 0 0 1 0 255 0 0 128 0.5
0 1 0 0 0 1 0 0 255 0 128 0.5
1 0 0 0 0 1 0 0 0 255 128 0.5
-1 0 0 0 0 1 0 255 255 0 128 0.5
3 0 1 2 0 255 0 255 128 0.5
3 0 1 3 0 255 0 255 128 0.5
```

 

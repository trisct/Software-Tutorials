# Trimesh Transformations

### Transformations

Common transformations can be obtained in the module `trimesh.transformations`. Usually a function in this module returns a transformation represented as a matrix. After obtaining the transformation, use `apply_transform`.

```
transform = trimesh.transformations.random_rotate_matrix()
mesh.apply_transform(transform)
```

The operation is inplace.
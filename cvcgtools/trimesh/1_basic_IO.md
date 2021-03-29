# Trimesh Basic IO

### Mesh Loading

Commonly used method for loading is `trimesh.load_mesh`.

```
mesh_obj = trimesh.load_mesh('mesh.obj')
mesh_stl = trimesh.load_mesh('mesh.stl')
mesh_stl = trimesh.load_mesh(file_obj='mesh.stl', file_type='stl')
```

### Mesh Exporting

```
mesh = trimesh.load_mesh('mesh.obj')
mesh.export('out_mesh_file.obj')
```

### Mesh Displaying

```
mesh.show()
```


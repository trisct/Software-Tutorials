# Trimesh Mesh Attributes

### Basic Attributes

Suppose `mesh` is a `trimesh.base.Trimesh` object.

- `mesh.vertices`
- `mesh.faces`
- `mesh.facets`

### Inquisitive Attributes

Some attributes answers geometric properties of the mesh.

- `mesh.is_watertight`
- `mesh.euler_number`
- `mesh.center_mass`
- `mesh.moment_inertia`

### Generative Attributes

Some attributes generate new geometric objects related to the current one.

- `mesh.convex_hull`
- `mesh.bounding_box`
- `mesh.bounding_box_oriented`
- `mesh.bounding_cylinder`
- `mesh.bounding_sphere`
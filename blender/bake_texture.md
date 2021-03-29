# Bake Texture

This one introduces specifically how to bake one OBJ, one MTL and multiple PNGs into one PNG texture.

### 1. Prepare and import files

Prepare the OBJ, MTL and PNG files and import with blender. Blender won't display texture in the default view but you can check under the _Shading_ tab.

To make things easier, the OBJ file should contain only one object (those grouped by `o` commands).

### 2. Duplicate object

Copy your object and delete all materials associated to the copy (remember to delete slots). Create one new material for the copy. Define a new UV map that unwraps the whole mesh to one.

Set a new _Image Texture_ node to the new material. Set a new _UV map_ node and connect it to the texture node.

### 3. Bake

Remember to choose the old UV map as the active one for rendering. Choose Cycles as the renderer and use the following options.

 - selected to active
 - ray distance > 0m
 - use only color influence.

### 4. Export

Since the UV map has changed, you must export the new mesh. Also remember to save the texture image.

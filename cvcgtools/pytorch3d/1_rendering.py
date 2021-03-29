# here you learn the following
# - loading a mesh
# - setting up a camera and a renderer
# - rendering

import os
import matplotlib.pyplot as plt


import torch
import pytorch3d
import pytorch3d.io as p3dio
import pytorch3d.structures as p3dstc
import pytorch3d.vis as p3dvis
import pytorch3d.renderer as p3drdr


### mesh loading
mesh_filename = 'mesh.obj'
# mesh = p3dio.load_obj(mesh_filename) # a tuple: (verts, faces, aux)
mesh = p3dio.load_objs_as_meshes([mesh_filename])


### camera, raster and renderer
R, T = look_at_view_transform(2.7, 0, 180)
cameras = FoVPerspectiveCameras(R=R, T=T)

raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        cameras=cameras,
        lights=lights
    )
)

images = rendered(mesh)

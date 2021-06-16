import pyrender
from pyrender import RenderFlags
import trimesh
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('sec_input', type=str)
args = parser.parse_args()


scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5],
                       bg_color=[1.0, 1.0, 1.0])

mesh_tri = trimesh.load_mesh(args.input)
mesh = pyrender.Mesh.from_trimesh(mesh_tri)

mesh_tri2 = trimesh.load_mesh(args.sec_input)
mesh2 = pyrender.Mesh.from_trimesh(mesh_tri2)

#light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.333)

nm = pyrender.Node(mesh=mesh, matrix=np.eye(4))
nm2 = pyrender.Node(mesh=mesh2, matrix=np.eye(4))

cam_pos = np.eye(4)
cam_pos[2, 3] = 1.
nc = pyrender.Node(camera=cam, matrix=cam_pos)
scene.add_node(nm)
scene.add_node(nm2)
#scene.add_node(nl)
scene.add_node(nc)

r = pyrender.OffscreenRenderer(viewport_width=640,
                               viewport_height=480,
                               point_size=0.1)

#color, depth = r.render(scene)
color, depth = r.render(scene, flags=RenderFlags.RGBA)
plt.imshow(color)
plt.show()

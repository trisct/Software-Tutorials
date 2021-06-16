import glfw
from OpenGL.GL import *
import numpy as np

# initializing glfw
glfw.init()

# creating a window with 800 width and 600 height
window = glfw.create_window(800,600,"PyOpenGL Triangle", None, None)
glfw.set_window_pos(window,400,200)
glfw.make_context_current(window)

vertices = [-0.5, -0.5, 0.0,
             0.5, -0.5,0.0,
             0.0, 0.5, 0.0]

v = np.array(vertices,dtype=np.float32)

# for drawing a colorless triangle
glEnableClientState(GL_VERTEX_ARRAY)
glVertexPointer(3, GL_FLOAT,0,v)

# setting color for background
glClearColor(0,0,0,0)

while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT)

    glDrawArrays(GL_TRIANGLES,0,3)
    glfw.swap_buffers(window)

glfw.terminate()

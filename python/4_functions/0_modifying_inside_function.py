import numpy as np

def change_inplace(pts):
    pts[:len(pts)//2] = 0

pts = np.random.randn(10)

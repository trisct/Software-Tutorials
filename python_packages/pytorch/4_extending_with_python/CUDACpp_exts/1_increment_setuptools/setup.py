from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='inc_cuda',
    ext_modules=[
        CUDAExtension('inc_cuda', [
            'inc_cuda.cpp',
            'inc_cuda.cu',
        ], include_dirs=['./'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

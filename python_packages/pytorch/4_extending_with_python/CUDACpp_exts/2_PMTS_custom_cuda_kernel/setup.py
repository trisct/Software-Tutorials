from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# [!!!] be sure to use different file names for cpp and cu files
# because `setuptools` does not see the filename extension

setup(
    name='PMTS_cuda',
    ext_modules=[
        CUDAExtension('PMTS_cuda', [
            'PMTS_cuda.cpp',
            'PMTS_cuda_kernels.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

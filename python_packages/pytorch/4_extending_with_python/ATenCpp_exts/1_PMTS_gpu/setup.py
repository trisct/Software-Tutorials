from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='PMTS_cpp',
      ext_modules=[cpp_extension.CppExtension('PMTS_cpp', ['PMTS.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
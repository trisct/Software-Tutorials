from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='inner_prod_cpp',
      ext_modules=[cpp_extension.CppExtension('inner_prod_cpp_ext', ['inner_prod.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})


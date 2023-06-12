from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='lltm_cpp',
    ext_modules=[
        CppExtension('lltm_cpp', ['lltm.cpp'], library_dirs=['/lib/x86_64-linux-gnu/'], runtime_library_dirs=['/lib/x86_64-linux-gnu/']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

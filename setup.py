# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)
# Conditional import for SyclExtension
try:
    from torch.utils.cpp_extension import SyclExtension
except ImportError:
    SyclExtension = None

library_name = "extension_cpp"

# Configure Py_LIMITED_API based on PyTorch version
if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False

def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"

    # Determine backend (CUDA, SYCL, or C++)
    use_cuda = os.getenv("USE_CUDA", "auto")
    use_sycl = os.getenv("USE_SYCL", "auto")

    # Auto-detect CUDA
    if use_cuda == "auto":
        use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    else:
        use_cuda = use_cuda.lower() == "true" or use_cuda == "1"

    # Auto-detect SYCL
    if use_sycl == "auto":
        use_sycl = SyclExtension is not None and torch.xpu.is_available()
    else:
        use_sycl = use_sycl.lower() == "true" or use_sycl == "1"

    if use_cuda and use_sycl:
        raise RuntimeError("Cannot enable both CUDA and SYCL backends simultaneously.")

    print("use cuda & use sycl",use_cuda, use_sycl)

    extension = None
    if use_cuda:
        extension = CUDAExtension
        print("Building with CUDA backend")
    elif use_sycl and SyclExtension is not None:
        extension = SyclExtension
        print("Building with SYCL backend")
    else:
        extension = CppExtension
        print("Building with C++ backend")

    # Compilation arguments
    extra_link_args = []
    extra_compile_args = {"cxx": []}
    if extension == CUDAExtension:
        extra_compile_args = {
            "cxx": ["-O3" if not debug_mode else "-O0",
                    "-fdiagnostics-color=always",
                    "-DPy_LIMITED_API=0x03090000"],
            "nvcc": ["-O3" if not debug_mode else "-O0"]
        }
    elif extension == SyclExtension:
        print("SYCLExtension branch, set extra_compile_args")
        extra_compile_args = {
            "cxx": ["-O3" if not debug_mode else "-O0",
                    "-fdiagnostics-color=always",
                    "-DPy_LIMITED_API=0x03090000"],
            "sycl": ["-O3" if not debug_mode else "-O0"]
        }
    else:
        extra_compile_args["cxx"] = [
            "-O3" if not debug_mode else "-O0",
            "-DPy_LIMITED_API=0x03090000"]

    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        if extension == CUDAExtension:
            extra_compile_args["nvcc"].append("-g")
            extra_link_args.extend(["-O0", "-g"])
        elif extension == SyclExtension:
            extra_compile_args["sycl"].append("-g")
            extra_link_args.extend(["-O0", "-g"])

    # Source files collection
    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    backend_sources = []
    if extension == CUDAExtension:
        backend_dir = os.path.join(extensions_dir, "cuda")
        backend_sources = glob.glob(os.path.join(backend_dir, "*.cu"))
    elif extension == SyclExtension:
        backend_dir = os.path.join(extensions_dir, "sycl")
        backend_sources = glob.glob(os.path.join(backend_dir, "*.sycl"))

    sources += backend_sources

    print("sources",sources)
    print(len(sources))
    # Construct extension
    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules

setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="Hybrid PyTorch extension supporting CUDA/SYCL/C++",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)

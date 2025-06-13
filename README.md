# C++/CUDA/SYCL Extensions in PyTorch

An example of writing a C++/CUDA/Sycl extension for PyTorch. See
[here](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html) for the accompanying tutorial.
This repo demonstrates how to write an example `extension_cpp.ops.mymuladd`
custom op that has both custom CPU and CUDA/Sycl kernels.


> **Note:**
  `SYCL` serves as the backend programming language for Intel GPUs (device label `xpu`). For configuration details, see:
  [Getting Started on Intel GPUs](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html).

The examples in this repo work with PyTorch 2.4 or later for C++/CUDA & PyTorch 2.8 or later for Sycl.

To build:
```
pip install --no-build-isolation -e .
```

To test:
```
python test/test_extension.py
```

## Authors

[Peter Goldsborough](https://github.com/goldsborough), [Richard Zou](https://github.com/zou3519)

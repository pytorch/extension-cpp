# C++/CUDA Extensions in PyTorch

This repository contains two example C++/CUDA extensions for PyTorch:

1. **extension_cpp** - Uses the standard ATen/LibTorch API
2. **extension_cpp_stable** - Uses the [LibTorch Stable ABI](https://pytorch.org/docs/main/notes/libtorch_stable_abi.html)

Both extensions demonstrate how to write an example `mymuladd` custom op that has both
custom CPU and CUDA kernels.

## extension_cpp (Standard ATen API)

Uses the full ATen/LibTorch API. This is the traditional way of writing PyTorch extensions.
See [this tutorial](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html) for more details.

## extension_cpp_stable (Stable ABI)

Uses the LibTorch Stable ABI to ensure that the extension built can be run with any version
of PyTorch >= 2.10.0, without needing to recompile for each PyTorch version.

The `extension_cpp_stable` examples require PyTorch 2.10+.

## Building

To build extension_cpp (standard API):
```
cd extension_cpp
pip install --no-build-isolation -e .
```

To build extension_cpp_stable (stable ABI):
```
cd extension_cpp_stable
pip install --no-build-isolation -e .
```

## Testing

To test both extensions:
```
python test/test_extension.py
```

## Authors

[Peter Goldsborough](https://github.com/goldsborough), [Richard Zou](https://github.com/zou3519)

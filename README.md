# C++/CUDA Extensions in PyTorch

An example of writing a C++/CUDA extension for PyTorch. See
[here](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html) for the accompanying tutorial.
This repo demonstrates how to write an example `extension_cpp.ops.mymuladd`
custom op that has both custom CPU and CUDA kernels.

The examples in this repo work with PyTorch 2.4+.

To build:
```
pip install .
```

To test:
```
python test/test_extension.py
```

To benchmark Python vs. C++ vs. CUDA:
```
python test/benchmark.py
```

## Authors

[Peter Goldsborough](https://github.com/goldsborough), [Richard Zou](https://github.com/zou3519)

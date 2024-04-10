# C++/CUDA Extensions in PyTorch

An example of writing a C++/CUDA extension for PyTorch. See
[here](http://pytorch.org/tutorials/advanced/cpp_extension.html) for the accompanying tutorial.

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

[Peter Goldsborough](https://github.com/goldsborough)

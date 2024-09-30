# C++/CUDA Extensions in PyTorch
# Table of Contents
1. [C++/CUDA Extensions in PyTorch](#ccuda-extensions-in-pytorch)
2. [Installation](#installation)
3. [To build](#to-build)
4. [To test](#to-test)
5. [To benchmark Python vs. C++ vs. CUDA](#to-benchmark-python-vs-c-vs-cuda)
6. [How to Contribute](#how-to-contribute)
    - [1. Fork the Repository](#1-fork-the-repository)
    - [2. Clone Your Fork](#2-clone-your-fork)
7. [Authors](#authors)
An example of writing a C++/CUDA extension for PyTorch. See
[here](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html) for the accompanying tutorial.
This repo demonstrates how to write an example `extension_cpp.ops.mymuladd`
custom op that has both custom CPU and CUDA kernels.

The examples in this repo work with PyTorch 2.4+.
## Installation
To use the following extension you must have the following:
- **PyTorch 2.4+**: Must have a compatible version installed.
- **CUDA Toolkit**: Required for building and running the CUDA kernels if using GPU acceleration.
- **GCC or Clang**: Necessary for compiling the C++ extension.
- **Python 3.7+**: Check you are using a compatible version of Python.

You can install the required packages using:
```bash
pip install torch
pip install -r requirements.txt  # If any additional requirements are specified
```
## To build:
```
pip install .
```

## To test:
```
python test/test_extension.py
```

## To benchmark Python vs. C++ vs. CUDA:
```
python test/benchmark.py
```
## How to Contribute
Contributions are always welcome, to contribute remember to do the following:
### 1. Fork the Repository
- Click the "Fork" button at the top of this repository to create your own copy.

### 2. Clone Your Fork
```bash
git clone https://github.com/<your-username>/extension-cpp.git
cd extension-cpp
```
## Authors

[Peter Goldsborough](https://github.com/goldsborough), [Richard Zou](https://github.com/zou3519)

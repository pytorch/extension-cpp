from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import glob
import torch
import torch.utils.cpp_extension
import pkg_resources

import python.lltm_baseline

def check_equal(first, second, verbose):
    if verbose:
        print()
    for i, (x, y) in enumerate(zip(first, second)):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        if verbose:
            print("x = {}".format(x.flatten()))
            print("y = {}".format(y.flatten()))
            print('-' * 80)
        np.testing.assert_allclose(x, y, rtol=2e-6, atol=2e-7, err_msg="Index: {}".format(i))

def zero_grad(variables):
    for variable in variables:
        variable.grad.zero_()


def get_grads(variables):
    return [var.grad.clone() for var in variables]


def check_forward(variables, with_cuda, verbose):
    baseline_values = python.lltm_baseline.LLTMFunction.apply(*variables)
    cpp_variables = [v.cpu() for v in variables]
    cpp_values = torch.ops.myops.lltm(*cpp_variables)

    print('Forward: Baseline (Python) vs. C++ ... ', end='')
    check_equal(baseline_values, cpp_values, verbose)
    print('Ok')

    if with_cuda:
        cuda_variables = [v.cuda() for v in variables]
        cuda_values = torch.ops.myops.lltm(*cuda_variables)
        print('Forward: Baseline (Python) vs. CUDA ... ', end='')
        check_equal(baseline_values, cuda_values, verbose)
        print('Ok')


def check_backward(variables, with_cuda, verbose):
    baseline_values = python.lltm_baseline.LLTMFunction.apply(*variables)
    (baseline_values[0] + baseline_values[1]).sum().backward()
    grad_baseline = get_grads(variables)

    zero_grad(variables)

    cpp_values = torch.ops.myops.lltm(*variables)
    (cpp_values[0] + cpp_values[1]).sum().backward()
    grad_cpp = get_grads(variables)

    print('Backward: Baseline (Python) vs. C++ ... ', end='')
    check_equal(grad_baseline, grad_cpp, verbose)
    print('Ok')

    if with_cuda:
        zero_grad(variables)
        cuda_values = torch.ops.myops.lltm(*variables)
        (cuda_values[0] + cuda_values[1]).sum().backward()
        grad_cuda = get_grads(variables)

        print('Backward: Baseline (Python) vs. CUDA ... ', end='')
        check_equal(grad_baseline, grad_cuda, verbose)
        print('Ok')


parser = argparse.ArgumentParser()
parser.add_argument('direction', choices=['forward', 'backward'], nargs='+')
parser.add_argument('-b', '--batch-size', type=int, default=3)
parser.add_argument('-f', '--features', type=int, default=17)
parser.add_argument('-s', '--state-size', type=int, default=5)
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-v', '--verbose', action='store_true')
options = parser.parse_args()

LIB_EXT = torch.utils.cpp_extension.LIB_EXT
cpp_module_path = os.path.dirname(
    pkg_resources.resource_filename(
        pkg_resources.Requirement.parse('lltm_cpp'), "lltm_cpp.py"))
cpp_lib_path = glob.glob(os.path.join(cpp_module_path, f"lltm_cpp*{LIB_EXT}"))[0]
torch.ops.load_library(cpp_lib_path)

if options.cuda:
    import cuda.lltm
    device = torch.device("cuda")

    cuda_module_path = os.path.dirname(
        pkg_resources.resource_filename(
            pkg_resources.Requirement.parse('lltm_cuda'), "lltm_cuda.py"))
    cuda_lib_path = glob.glob(os.path.join(cuda_module_path, f"lltm_cuda*{LIB_EXT}"))[0]
    torch.ops.load_library(cuda_lib_path)
else:
    device = torch.device("cpu")

kwargs = {'dtype': torch.float64,
          'device': device,
          'requires_grad': True}
X = torch.randn(options.batch_size,
                options.features,
                **kwargs)
h = torch.randn(options.batch_size, options.state_size, **kwargs)
C = torch.randn(options.batch_size, options.state_size, **kwargs)
W = torch.randn(3 * options.state_size, options.features + options.state_size, **kwargs)
b = torch.randn(1, 3 * options.state_size, **kwargs)

variables = [X, W, b, h, C]


if 'forward' in options.direction:
    check_forward(variables, options.cuda, options.verbose)

if 'backward' in options.direction:
    check_backward(variables, options.cuda, options.verbose)

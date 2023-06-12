from __future__ import division
from __future__ import print_function

import argparse
import os
import pkg_resources
import torch
from torch.autograd import gradcheck
import glob

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cpp', 'cuda'])
parser.add_argument('-b', '--batch-size', type=int, default=3)
parser.add_argument('-f', '--features', type=int, default=17)
parser.add_argument('-s', '--state-size', type=int, default=5)
parser.add_argument('-c', '--cuda', action='store_true')
options = parser.parse_args()

cpp_module_path = os.path.dirname(
    pkg_resources.resource_filename(
        pkg_resources.Requirement.parse('lltm_cpp'), "lltm_cpp.py"))
cpp_lib_path = glob.glob(os.path.join(cpp_module_path, "lltm_cpp*.so"))[0]
torch.ops.load_library(cpp_lib_path)

cuda_module_path = os.path.dirname(
    pkg_resources.resource_filename(
        pkg_resources.Requirement.parse('lltm_cuda'), "lltm_cuda.py"))
cuda_lib_path = glob.glob(os.path.join(cuda_module_path, "lltm_cuda*.so"))[0]
torch.ops.load_library(cuda_lib_path)


if options.example == 'py':
    from python.lltm_baseline import LLTMFunction
    lltm_func = LLTMFunction.apply
else:
    lltm_func = torch.ops.myops.lltm

options.cuda |= (options.example == "cuda")

device = torch.device("cuda") if options.cuda else torch.device("cpu")

kwargs = {'dtype': torch.float64,
          'device': device,
          'requires_grad': True}

X = torch.randn(options.batch_size, options.features, **kwargs)
h = torch.randn(options.batch_size, options.state_size, **kwargs)
C = torch.randn(options.batch_size, options.state_size, **kwargs)
W = torch.randn(3 * options.state_size,
                options.features + options.state_size,
                **kwargs)
b = torch.randn(1, 3 * options.state_size, **kwargs)

variables = [X, W, b, h, C]


if gradcheck(lltm_func, variables):
    print('Ok')

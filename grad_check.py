from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.autograd import gradcheck

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cpp', 'cuda'])
parser.add_argument('-b', '--batch-size', type=int, default=3)
parser.add_argument('-f', '--features', type=int, default=17)
parser.add_argument('-s', '--state-size', type=int, default=5)
parser.add_argument('-c', '--cuda', action='store_true')
options = parser.parse_args()

torch.ops.load_library("cpp/build/lib.linux-x86_64-cpython-39/lltm_cpp.cpython-39-x86_64-linux-gnu.so")
torch.ops.load_library("cuda/build/lib.linux-x86_64-cpython-39/lltm_cuda.cpython-39-x86_64-linux-gnu.so")

if options.example == 'py':
    from python.lltm_baseline import LLTMFunction
elif options.example == 'cpp':
    from cpp.lltm import LLTMFunction
else:
    from cuda.lltm import LLTMFunction
    options.cuda = True

device = torch.device("cuda") if options.cuda else torch.device("cpu")

kwargs = {'dtype': torch.float64,
          'device': device,
          'requires_grad': True}

X = torch.randn(options.batch_size, options.features, **kwargs).to(device)
h = torch.randn(options.batch_size, options.state_size, **kwargs).to(device)
C = torch.randn(options.batch_size, options.state_size, **kwargs).to(device)
W = torch.randn(3 * options.state_size, options.features + options.state_size, **kwargs).to(device)
b = torch.randn(1, 3 * options.state_size, **kwargs).to(device)

variables = [X, W, b, h, C]


if gradcheck(torch.ops.myops.lltm, variables):
    print('Ok')

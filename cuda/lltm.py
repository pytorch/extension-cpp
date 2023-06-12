import math
import os
from torch import nn
from torch.autograd import Function
import torch
import glob
import torch.utils.cpp_extension
import pkg_resources

# Get the location of shared library for the lltm op, and load it.
LIB_EXT = torch.utils.cpp_extension.LIB_EXT
# Note: currently there is a dependency on the CPP lib, due to the schema definition
# Eventually, this should move to use a single library registering both CPP and CUDA ops
cpp_module_path = os.path.dirname(
    pkg_resources.resource_filename(
        pkg_resources.Requirement.parse('lltm_cpp'), "lltm_cpp.py"))
cpp_lib_path = glob.glob(os.path.join(cpp_module_path, f"lltm_cpp*{LIB_EXT}"))[0]
torch.ops.load_library(cpp_lib_path)
cuda_module_path = os.path.dirname(
    pkg_resources.resource_filename(
        pkg_resources.Requirement.parse('lltm_cuda'), "lltm_cuda.py"))
cuda_lib_path = glob.glob(os.path.join(cuda_module_path, f"lltm_cuda*{LIB_EXT}"))[0]
torch.ops.load_library(cuda_lib_path)

torch.manual_seed(42)

class LLTM(nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = nn.Parameter(
            torch.Tensor(3 * state_size, input_features + state_size))
        self.bias = nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return torch.ops.myops.lltm(input, self.weights, self.bias, *state)

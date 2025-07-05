#pragma once
#include <torch/torch.h>

namespace extension_cpp {

TORCH_API torch::Tensor fast_sigmoid(
    const torch::Tensor& input,
    double min = -10.0,
    double max = 10.0,
    int64_t num_entries = 1000
);

} // namespace extension_cpp

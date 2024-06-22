#pragma once

#include <torch/torch.h>

namespace driss_torch {

at::Tensor saturated_cast(const at::Tensor &input, const at::Tensor &attn_mask,
                          at::ScalarType dtype, bool transpose);
at::Tensor saturated_cast_meta(const at::Tensor &input,
                               const at::Tensor &attn_mask,
                               at::ScalarType dtype, bool transpose);
} // namespace driss_torch

#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> lltm_cuda_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell);

std::vector<at::Tensor> lltm_cuda_backward(
    at::Tensor grad_h,
    at::Tensor grad_cell,
    at::Tensor new_cell,
    at::Tensor input_gate,
    at::Tensor output_gate,
    at::Tensor candidate_cell,
    at::Tensor X,
    at::Tensor gate_weights,
    at::Tensor weights);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")

std::vector<at::Tensor> lltm_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell) {
  CHECK_CUDA(input);
  CHECK_CUDA(weights);
  CHECK_CUDA(bias);
  CHECK_CUDA(old_h);
  CHECK_CUDA(old_cell);

  return lltm_cuda_forward(input, weights, bias, old_h, old_cell);
}

std::vector<at::Tensor> lltm_backward(
    at::Tensor grad_h,
    at::Tensor grad_cell,
    at::Tensor new_cell,
    at::Tensor input_gate,
    at::Tensor output_gate,
    at::Tensor candidate_cell,
    at::Tensor X,
    at::Tensor gate_weights,
    at::Tensor weights) {
  CHECK_CUDA(grad_h);
  CHECK_CUDA(grad_cell);
  CHECK_CUDA(input_gate);
  CHECK_CUDA(output_gate);
  CHECK_CUDA(candidate_cell);
  CHECK_CUDA(X);
  CHECK_CUDA(gate_weights);
  CHECK_CUDA(weights);

  return lltm_cuda_backward(
      grad_h,
      grad_cell,
      new_cell,
      input_gate,
      output_gate,
      candidate_cell,
      X,
      gate_weights,
      weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward (CUDA)");
  m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}

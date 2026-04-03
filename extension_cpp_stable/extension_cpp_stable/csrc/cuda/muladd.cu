#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <torch/csrc/stable/c/shim.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace extension_cpp_stable {

__global__ void muladd_kernel(int numel, const float *a, const float *b,
                              float c, float *result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel)
    result[idx] = a[idx] * b[idx] + c;
}

torch::stable::Tensor mymuladd_cuda(const torch::stable::Tensor &a,
                                    const torch::stable::Tensor &b, double c) {
  STD_TORCH_CHECK(a.sizes().equals(b.sizes()));
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CUDA);
  STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CUDA);

  torch::stable::Tensor a_contig = torch::stable::contiguous(a);
  torch::stable::Tensor b_contig = torch::stable::contiguous(b);
  torch::stable::Tensor result = torch::stable::empty_like(a_contig);

  const float *a_ptr = a_contig.const_data_ptr<float>();
  const float *b_ptr = b_contig.const_data_ptr<float>();
  float *result_ptr = result.mutable_data_ptr<float>();

  int numel = a_contig.numel();

  // For now, we rely on the raw shim API to get the current CUDA stream.
  // This will be improved in a future release.
  // When using a raw shim API, we need to use TORCH_ERROR_CODE_CHECK to
  // check the error code and throw an appropriate runtime_error otherwise.
  void *stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(a.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  muladd_kernel<<<(numel + 255) / 256, 256, 0, stream>>>(numel, a_ptr, b_ptr, c,
                                                         result_ptr);
  return result;
}

__global__ void mul_kernel(int numel, const float *a, const float *b,
                           float *result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel)
    result[idx] = a[idx] * b[idx];
}

torch::stable::Tensor mymul_cuda(const torch::stable::Tensor &a,
                                 const torch::stable::Tensor &b) {
  STD_TORCH_CHECK(a.sizes().equals(b.sizes()));
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CUDA);
  STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CUDA);

  torch::stable::Tensor a_contig = torch::stable::contiguous(a);
  torch::stable::Tensor b_contig = torch::stable::contiguous(b);
  torch::stable::Tensor result = torch::stable::empty_like(a_contig);

  const float *a_ptr = a_contig.const_data_ptr<float>();
  const float *b_ptr = b_contig.const_data_ptr<float>();
  float *result_ptr = result.mutable_data_ptr<float>();

  int numel = a_contig.numel();

  // For now, we rely on the raw shim API to get the current CUDA stream.
  // This will be improved in a future release.
  // When using a raw shim API, we need to use TORCH_ERROR_CODE_CHECK to
  // check the error code and throw an appropriate runtime_error otherwise.
  void *stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(a.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  mul_kernel<<<(numel + 255) / 256, 256, 0, stream>>>(numel, a_ptr, b_ptr,
                                                      result_ptr);
  return result;
}

__global__ void add_kernel(int numel, const float *a, const float *b,
                           float *result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel)
    result[idx] = a[idx] + b[idx];
}

// An example of an operator that mutates one of its inputs.
void myadd_out_cuda(const torch::stable::Tensor &a,
                    const torch::stable::Tensor &b,
                    torch::stable::Tensor &out) {
  STD_TORCH_CHECK(a.sizes().equals(b.sizes()));
  STD_TORCH_CHECK(b.sizes().equals(out.sizes()));
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(out.is_contiguous());
  STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CUDA);
  STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CUDA);
  STD_TORCH_CHECK(out.device().type() == torch::headeronly::DeviceType::CUDA);

  torch::stable::Tensor a_contig = torch::stable::contiguous(a);
  torch::stable::Tensor b_contig = torch::stable::contiguous(b);

  const float *a_ptr = a_contig.const_data_ptr<float>();
  const float *b_ptr = b_contig.const_data_ptr<float>();
  float *result_ptr = out.mutable_data_ptr<float>();

  int numel = a_contig.numel();

  // For now, we rely on the raw shim API to get the current CUDA stream.
  // This will be improved in a future release.
  // When using a raw shim API, we need to use TORCH_ERROR_CODE_CHECK to
  // check the error code and throw an appropriate runtime_error otherwise.
  void *stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(a.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  add_kernel<<<(numel + 255) / 256, 256, 0, stream>>>(numel, a_ptr, b_ptr,
                                                      result_ptr);
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
STABLE_TORCH_LIBRARY_IMPL(extension_cpp_stable, CUDA, m) {
  m.impl("mymuladd", TORCH_BOX(&mymuladd_cuda));
  m.impl("mymul", TORCH_BOX(&mymul_cuda));
  m.impl("myadd_out", TORCH_BOX(&myadd_out_cuda));
}

} // namespace extension_cpp_stable

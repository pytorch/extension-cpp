#include <Python.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the STABLE_TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace extension_cpp_stable {

torch::stable::Tensor mymuladd_cpu(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    double c) {
  STD_TORCH_CHECK(a.sizes().equals(b.sizes()));
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CPU);
  STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CPU);

  torch::stable::Tensor a_contig = torch::stable::contiguous(a);
  torch::stable::Tensor b_contig = torch::stable::contiguous(b);
  torch::stable::Tensor result = torch::stable::empty_like(a_contig);

  const float* a_ptr = a_contig.const_data_ptr<float>();
  const float* b_ptr = b_contig.const_data_ptr<float>();
  float* result_ptr = result.mutable_data_ptr<float>();

  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
  }
  return result;
}

torch::stable::Tensor mymul_cpu(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b) {
  STD_TORCH_CHECK(a.sizes().equals(b.sizes()));
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CPU);
  STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CPU);

  torch::stable::Tensor a_contig = torch::stable::contiguous(a);
  torch::stable::Tensor b_contig = torch::stable::contiguous(b);
  torch::stable::Tensor result = torch::stable::empty_like(a_contig);

  const float* a_ptr = a_contig.const_data_ptr<float>();
  const float* b_ptr = b_contig.const_data_ptr<float>();
  float* result_ptr = result.mutable_data_ptr<float>();

  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i];
  }
  return result;
}

// An example of an operator that mutates one of its inputs.
void myadd_out_cpu(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    torch::stable::Tensor& out) {
  STD_TORCH_CHECK(a.sizes().equals(b.sizes()));
  STD_TORCH_CHECK(b.sizes().equals(out.sizes()));
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(out.is_contiguous());
  STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CPU);
  STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CPU);
  STD_TORCH_CHECK(out.device().type() == torch::headeronly::DeviceType::CPU);

  torch::stable::Tensor a_contig = torch::stable::contiguous(a);
  torch::stable::Tensor b_contig = torch::stable::contiguous(b);

  const float* a_ptr = a_contig.const_data_ptr<float>();
  const float* b_ptr = b_contig.const_data_ptr<float>();
  float* result_ptr = out.mutable_data_ptr<float>();

  for (int64_t i = 0; i < out.numel(); i++) {
    result_ptr[i] = a_ptr[i] + b_ptr[i];
  }
}

// Defines the operators
STABLE_TORCH_LIBRARY(extension_cpp_stable, m) {
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
  m.def("mymul(Tensor a, Tensor b) -> Tensor");
  m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
}

// Registers CPU implementations for mymuladd, mymul, myadd_out
STABLE_TORCH_LIBRARY_IMPL(extension_cpp_stable, CPU, m) {
  m.impl("mymuladd", TORCH_BOX(&mymuladd_cpu));
  m.impl("mymul", TORCH_BOX(&mymul_cpu));
  m.impl("myadd_out", TORCH_BOX(&myadd_out_cpu));
}

}

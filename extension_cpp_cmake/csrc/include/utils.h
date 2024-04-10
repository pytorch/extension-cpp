#pragma once

#include <functional>
#include <stdio.h>


namespace driss_torch {

template <typename... Args>
__device__ void thread_zero_print(const char *fmt, Args &&...args) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf(fmt, std::forward<Args>(args)...);
  }
}

// error checking macro
#define cudaCheckErrors(msg)                                                   \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,                  \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      fprintf(stderr, "*** FAILED - ABORTING\n");                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

template <typename T, typename Y> T __host__ __device__ ceil_div(T a, Y b) {
  return a / b + (a % b != 0);
}
// Functions whose implementation is in the .cu file
extern "C" {

float kernel_time(std::function<void()> kernelLauncher);

} // extern "C"

} // namespace driss_torch

#ifndef HSPLAT_CUDA_BINDINGS_H
#define HSPLAT_CUDA_BINDINGS_H

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <tuple>

#define PI 3.14159265358979323846f  // Single precision

#define HSPLAT_N_THREADS 256

#define HSPLAT_CHECK_CUDA(x)                                                   \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define HSPLAT_CHECK_CONTIGUOUS(x)                                             \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define HSPLAT_CHECK_INPUT(x)                                                  \
    HSPLAT_CHECK_CUDA(x);                                                      \
    HSPLAT_CHECK_CONTIGUOUS(x)
#define HSPLAT_DEVICE_GUARD(_ten)                                              \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

#define HSPLAT_PRAGMA_UNROLL _Pragma("unroll")

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
#define HSPLAT_CUB_WRAPPER(func, ...)                                          \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);             \
    } while (false)

namespace hsplat {

// simple CUDA test function
torch::Tensor add_tensor(
    const torch::Tensor &a, // [N, 3]
    const torch::Tensor &b  // [N, 3]
);

/*
    Fast CUDA CGH implementations
*/

std::tuple<torch::Tensor, torch::Tensor> cgh_gaussians_naive_tensor(
    const torch::Tensor &fx,             // [2*H, 2*W],
    const torch::Tensor &fy,             // [2*H, 2*W],
    const torch::Tensor &fz,             // [2*H, 2*W],
    const float         wvl,             
    const torch::Tensor &R,              // [N, 3, 3],
    const torch::Tensor &A_inv_T,        // [N, 2, 2],
    const torch::Tensor &A_det,          // [N],
    const torch::Tensor &c,              // [N, 3]
    const torch::Tensor &du,             // [N, 3],
    const torch::Tensor &local_AS_shift, // [N]
    const torch::Tensor &opacity,        // [N]
    const torch::Tensor &color           // [N]
);

} // namespace hsplat

#endif // HSPLAT_CUDA_BINDINGS_H

#include "bindings.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace hsplat {

namespace cg = cooperative_groups;

template <typename T>
__global__ void add_kernel(
    const uint32_t N,
    const T *__restrict__ a,  // [N, 3]
    const T *__restrict__ b,  // [N, 3]
    T *__restrict__ c         // [N, 3]
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N * 3) {
        return;
    }
    c[idx] = a[idx] + b[idx];
}

torch::Tensor add_tensor(
    const torch::Tensor &a,  // [N, 3]
    const torch::Tensor &b   // [N, 3]
) {
    HSPLAT_DEVICE_GUARD(a);
    HSPLAT_CHECK_INPUT(a);
    HSPLAT_CHECK_INPUT(b);

    // Validate tensor dimensions
    TORCH_CHECK(a.size(-1) == 3, "Input tensor 'a' must have size[-1] == 3");
    TORCH_CHECK(b.size(-1) == 3, "Input tensor 'b' must have size[-1] == 3");
    TORCH_CHECK(a.size(0) == b.size(0), "Input tensors must have the same number of rows");

    uint32_t N = static_cast<uint32_t>(a.size(0));

    torch::Tensor c = torch::empty({N, 3}, a.options());
    if (N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            a.scalar_type(),
            "add_tensor",
            [&]() {
                add_kernel<scalar_t>
                    <<< (N * 3 + HSPLAT_N_THREADS - 1) / HSPLAT_N_THREADS,
                        HSPLAT_N_THREADS, 0, stream >>>(
                        N,
                        a.data_ptr<scalar_t>(),
                        b.data_ptr<scalar_t>(),
                        c.data_ptr<scalar_t>()
                    );
            }
        );
    }
    return c;
}

} // namespace hsplat

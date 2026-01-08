#include "bindings.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace hsplat {

namespace cg = cooperative_groups;

template <typename T>
__global__ void cgh_gaussians_naive_kernel(
    const uint32_t        N,
    const uint32_t        grid_H,
    const uint32_t        grid_W,
    const T *__restrict__ fx,             // [2*H, 2*W],
    const T *__restrict__ fy,             // [2*H, 2*W],
    const T *__restrict__ fz,             // [2*H, 2*W],
    const float           wvl,
    const T *__restrict__ R,              // [N, 3, 3],
    const T *__restrict__ A_inv_T,        // [N, 2, 2],
    const T *__restrict__ A_det,          // [N],
    const T *__restrict__ c,              // [N, 3]
    const T *__restrict__ du,             // [N, 3],
    const T *__restrict__ local_AS_shift, // [N]    
    const T *__restrict__ opacity,        // [N]
    const T *__restrict__ colors,         // [N]
    T *__restrict__       G_real,         // [2*H, 2*W]
    T *__restrict__       G_imag          // [2*H, 2*W]
) {
    // Calculate the global pixel indices
    const uint32_t pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t threadIdx1D = threadIdx.y * blockDim.x + threadIdx.x;

    // Boundary check for pixels
    if (pixel_x >= grid_W || pixel_y >= grid_H) {
        return;
    }

    const uint32_t pixel_id = pixel_y * grid_W + pixel_x; // flatten 2D pixel indices
    // Original spectrum values
    T fx_val = fx[pixel_id];
    T fy_val = fy[pixel_id];  
    T fz_val = fz[pixel_id];
    // Results
    T G_real_val = 0.0; // Accumulated real part of G
    T G_imag_val = 0.0; // Accumulated imaginary part of G
    
    const uint32_t batch_size = 200; // Number of Gaussians to load into shared memory
    // Shared Gaussian parameters and arrays
    __shared__ T shared_GS_params[batch_size * (9 + 4 + 1 + 3 + 3 + 1 + 1 + 1)]; // 11960 * sizeof(float) = 47840 bytes
    T *shared_R = shared_GS_params;
    T *shared_A_inv_T = shared_GS_params + batch_size * 9;
    T *shared_A_det = shared_GS_params + batch_size * (9 + 4);
    T *shared_c = shared_GS_params + batch_size * (9 + 4 + 1);
    T *shared_du = shared_GS_params + batch_size * (9 + 4 + 1 + 3);
    T *shared_local_AS_shift = shared_GS_params + batch_size * (9 + 4 + 1 + 3 + 3);
    T *shared_opacity = shared_GS_params + batch_size * (9 + 4 + 1 + 3 + 3 + 1);
    T *shared_colors = shared_GS_params + batch_size * (9 + 4 + 1 + 3 + 3 + 1 + 1);

    for(uint32_t gaussian_start_idx = 0; gaussian_start_idx < N; gaussian_start_idx += batch_size) {
        // Load Gaussian parameters into shared memory
        __syncthreads();
        if(threadIdx1D < batch_size) {
            // Each thread loads parameters for a single Gaussian
            uint32_t gaussian_id = gaussian_start_idx + threadIdx1D;
            if(gaussian_id < N) {
                HSPLAT_PRAGMA_UNROLL
                for (int i = 0; i < 9; ++i) shared_R[threadIdx1D * 9 + i] = R[gaussian_id * 9 + i];
                HSPLAT_PRAGMA_UNROLL
                for (int i = 0; i < 4; ++i) shared_A_inv_T[threadIdx1D * 4 + i] = A_inv_T[gaussian_id * 4 + i];
                shared_A_det[threadIdx1D] = A_det[gaussian_id];
                HSPLAT_PRAGMA_UNROLL
                for (int i = 0; i < 3; ++i) {
                    shared_c[threadIdx1D * 3 + i] = c[gaussian_id * 3 + i];
                    shared_du[threadIdx1D * 3 + i] = du[gaussian_id * 3 + i];
                }
                shared_local_AS_shift[threadIdx1D] = local_AS_shift[gaussian_id];
                shared_opacity[threadIdx1D] = opacity[gaussian_id];
                shared_colors[threadIdx1D] = colors[gaussian_id];
            }
        }
        __syncthreads();
        // Now process each Gaussian
        for(uint32_t offset = 0; offset < batch_size; ++offset) {
            uint32_t gaussian_id = gaussian_start_idx + offset;
            if(gaussian_id < N) {
                // Rotate from world -> local
                T flx_val = fx_val * shared_R[offset * 9 + 0] + fy_val * shared_R[offset * 9 + 1] + fz_val * shared_R[offset * 9 + 2];
                T fly_val = fx_val * shared_R[offset * 9 + 3] + fy_val * shared_R[offset * 9 + 4] + fz_val * shared_R[offset * 9 + 5];
                T flz_val = sqrtf(1.0 / (wvl * wvl) - flx_val * flx_val - fly_val * fly_val);
                flz_val = fmaxf(flz_val, 1e-12);

                // Carrier wave
                T fx_l_offset_val = flx_val - shared_du[offset * 3 + 0];
                T fy_l_offset_val = fly_val - shared_du[offset * 3 + 1];
                
                // Rotate from local -> reference
                T fx_ref_val = fx_l_offset_val * shared_A_inv_T[offset * 4 + 0] + fy_l_offset_val * shared_A_inv_T[offset * 4 + 1];
                T fy_ref_val = fx_l_offset_val * shared_A_inv_T[offset * 4 + 2] + fy_l_offset_val * shared_A_inv_T[offset * 4 + 3];

                // Angular spectrum of Gaussian in reference coordinates
                T G0 = 2 * PI * expf(-2.0 * PI * PI * (fx_ref_val * fx_ref_val + fy_ref_val * fy_ref_val));

                // Collapse reference -> local and local -> world into single operation to minimize complex numbers computation
                T phi = 2 * PI * (flx_val * shared_c[offset * 3 + 0] + 
                                fly_val * shared_c[offset * 3 + 1] + 
                                flz_val * shared_c[offset * 3 + 2]) -
                        2 * PI / wvl * shared_local_AS_shift[offset];
                T scale = (flz_val / fz_val) * (1.0 / (shared_A_det[offset]));

                G_real_val += G0 * scale * cosf(phi) * shared_opacity[offset] * shared_colors[offset];
                G_imag_val += G0 * scale * sinf(phi) * shared_opacity[offset] * shared_colors[offset];
            }
        }
    }
    G_real[pixel_id] = G_real_val;
    G_imag[pixel_id] = G_imag_val;
}

std::tuple<torch::Tensor, torch::Tensor> cgh_gaussians_naive_tensor(
    const torch::Tensor &fx,             // [2*H, 2*W],
    const torch::Tensor &fy,             // [2*H, 2*W],
    const torch::Tensor &fz,             // [2*H, 2*W],
    const float         wvl,            
    const torch::Tensor &R,              // [N, 3, 3], 9
    const torch::Tensor &A_inv_T,        // [N, 2, 2], 4
    const torch::Tensor &A_det,          // [N], 1
    const torch::Tensor &c,              // [N, 3], 3
    const torch::Tensor &du,             // [N, 3], 3
    const torch::Tensor &local_AS_shift, // [N], 1
    const torch::Tensor &opacity,        // [N], 1
    const torch::Tensor &colors          // [N], 1
) {
    // CUDA checks
    HSPLAT_DEVICE_GUARD(fx);
    HSPLAT_CHECK_INPUT(fx);
    HSPLAT_CHECK_INPUT(fy);
    HSPLAT_CHECK_INPUT(fz);
    HSPLAT_CHECK_INPUT(R);
    HSPLAT_CHECK_INPUT(A_inv_T);
    HSPLAT_CHECK_INPUT(A_det);
    HSPLAT_CHECK_INPUT(c);
    HSPLAT_CHECK_INPUT(du);
    HSPLAT_CHECK_INPUT(local_AS_shift);
    HSPLAT_CHECK_INPUT(opacity);
    HSPLAT_CHECK_INPUT(colors);

    uint32_t N = static_cast<uint32_t>(R.size(0)); // Number of Gaussians
    uint32_t grid_H = static_cast<uint32_t>(fx.size(0)); // spectrum grid height 
    uint32_t grid_W = static_cast<uint32_t>(fx.size(1)); // spectrum grid width

    dim3 threads_per_block(32, 32); // 32 * 32 = 1024 threads per block
    dim3 num_blocks(
        (grid_W + threads_per_block.x - 1) / threads_per_block.x,
        (grid_H + threads_per_block.y - 1) / threads_per_block.y
    );
    printf("num_blocks: [%d, %d]\n", num_blocks.x, num_blocks.y);
    // Allocate shared memory for Gaussian parameters and intermediate pixel contributions
    size_t shared_mem_size = (200 * (9 + 4 + 1 + 3 + 3 + 1 + 1 + 1)) * sizeof(float); // need to test max. shared memory. Maxium is 48KB but still platform dependent

    // Check for device memory and CUDA limits
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    printf("Max threads per block: %d\n", maxThreadsPerBlock);

    printf("Shared memory size: %zu bytes\n", shared_mem_size);

    // Ensure shared memory doesn't exceed device limits
    int sharedMemPerBlock;
    cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (shared_mem_size > sharedMemPerBlock) {
        printf("Error: Requested shared memory size exceeds device limit!\n");
        return std::make_tuple(torch::empty({0}), torch::empty({0}));
    }

    torch::Tensor G_real = torch::zeros({grid_H, grid_W}, fx.options()); // real part of CGH spectrum
    torch::Tensor G_imag = torch::zeros({grid_H, grid_W}, fx.options()); // imaginary part of CGH spectrum
    if (N > 0) {
        // at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES(
            fx.scalar_type(),
            "cgh_gaussians_naive_tensor",
            [&]() {
                cgh_gaussians_naive_kernel<scalar_t>
                    <<<num_blocks, threads_per_block, shared_mem_size >>>(
                        N,
                        grid_H,
                        grid_W,
                        fx.data_ptr<scalar_t>(),
                        fy.data_ptr<scalar_t>(),
                        fz.data_ptr<scalar_t>(),
                        wvl,
                        R.data_ptr<scalar_t>(),
                        A_inv_T.data_ptr<scalar_t>(),
                        A_det.data_ptr<scalar_t>(),
                        c.data_ptr<scalar_t>(),
                        du.data_ptr<scalar_t>(),
                        local_AS_shift.data_ptr<scalar_t>(),
                        opacity.data_ptr<scalar_t>(),
                        colors.data_ptr<scalar_t>(),
                        G_real.data_ptr<scalar_t>(),
                        G_imag.data_ptr<scalar_t>()
                    );
            }
        );

        // Kernel launch error check
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(error));
        }

        // Synchronize and check for errors
        cudaDeviceSynchronize();
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error after synchronization: %s\n", cudaGetErrorString(error));
        }
    }
    return std::make_tuple(G_real, G_imag);
}

} // namespace hsplat

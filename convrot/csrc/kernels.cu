#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

template <typename scalar_t>
__global__ void hadamard_rotate_kernel_general(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t batch,
    int64_t k,
    int64_t rot_size,
    int64_t num_blocks,
    float inv_norm) {
    int64_t row = blockIdx.x;
    int64_t block_id = blockIdx.y;
    int64_t tid = threadIdx.x;
    if (row >= batch || block_id >= num_blocks || tid >= rot_size) {
        return;
    }

    int64_t base = block_id * rot_size;
    int64_t col = base + tid;

    extern __shared__ float smem[];

    if (col < k) {
        smem[tid] = static_cast<float>(input[row * k + col]);
    } else {
        smem[tid] = 0.0f;
    }
    __syncthreads();

    for (int stride = 1; stride < rot_size; stride *= 4) {
        int group_span = stride * 4;
        int group = static_cast<int>(tid) / group_span;
        int lane = static_cast<int>(tid) % stride;
        int quad = (static_cast<int>(tid) / stride) & 0x3;

        if (quad == 0) {
            int i0 = group * group_span + lane;
            int i1 = i0 + stride;
            int i2 = i0 + 2 * stride;
            int i3 = i0 + 3 * stride;

            float a = smem[i0];
            float b = smem[i1];
            float c = smem[i2];
            float d = smem[i3];

            smem[i0] = a + b + c - d;
            smem[i1] = a + b - c + d;
            smem[i2] = a - b + c + d;
            smem[i3] = -a + b + c + d;
        }
        __syncthreads();
    }

    if (col < k) {
        output[row * k + col] = static_cast<scalar_t>(smem[tid] * inv_norm);
    }
}

template <int ROT_SIZE, bool FULL, typename scalar_t>
__global__ void hadamard_rotate_kernel_static(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t batch,
    int64_t k,
    int64_t num_blocks,
    float inv_norm) {
    int64_t row = blockIdx.x;
    int64_t block_id = blockIdx.y;
    int64_t tid = threadIdx.x;
    if (row >= batch || tid >= ROT_SIZE) {
        return;
    }
    if (!FULL && block_id >= num_blocks) {
        return;
    }

    int64_t base = block_id * ROT_SIZE;
    int64_t col = base + tid;

    extern __shared__ float smem[];
    if (FULL) {
        smem[tid] = static_cast<float>(input[row * k + col]);
    } else {
        if (col < k) {
            smem[tid] = static_cast<float>(input[row * k + col]);
        } else {
            smem[tid] = 0.0f;
        }
    }
    __syncthreads();

#pragma unroll
    for (int stride = 1; stride < ROT_SIZE; stride *= 4) {
        int group_span = stride * 4;
        int group = static_cast<int>(tid) / group_span;
        int lane = static_cast<int>(tid) % stride;
        int quad = (static_cast<int>(tid) / stride) & 0x3;

        if (quad == 0) {
            int i0 = group * group_span + lane;
            int i1 = i0 + stride;
            int i2 = i0 + 2 * stride;
            int i3 = i0 + 3 * stride;

            float a = smem[i0];
            float b = smem[i1];
            float c = smem[i2];
            float d = smem[i3];

            smem[i0] = a + b + c - d;
            smem[i1] = a + b - c + d;
            smem[i2] = a - b + c + d;
            smem[i3] = -a + b + c + d;
        }
        __syncthreads();
    }

    if (FULL || col < k) {
        output[row * k + col] = static_cast<scalar_t>(smem[tid] * inv_norm);
    }
}

template <typename scalar_t>
__global__ void hadamard_rotate_kernel_full(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t batch,
    int64_t k,
    int64_t rot_size,
    float inv_norm) {
    int64_t row = blockIdx.x;
    int64_t block_id = blockIdx.y;
    int64_t tid = threadIdx.x;
    if (row >= batch || tid >= rot_size) {
        return;
    }

    int64_t base = block_id * rot_size;
    int64_t col = base + tid;

    extern __shared__ float smem[];
    smem[tid] = static_cast<float>(input[row * k + col]);
    __syncthreads();

    for (int stride = 1; stride < rot_size; stride *= 4) {
        int group_span = stride * 4;
        int group = static_cast<int>(tid) / group_span;
        int lane = static_cast<int>(tid) % stride;
        int quad = (static_cast<int>(tid) / stride) & 0x3;

        if (quad == 0) {
            int i0 = group * group_span + lane;
            int i1 = i0 + stride;
            int i2 = i0 + 2 * stride;
            int i3 = i0 + 3 * stride;

            float a = smem[i0];
            float b = smem[i1];
            float c = smem[i2];
            float d = smem[i3];

            smem[i0] = a + b + c - d;
            smem[i1] = a + b - c + d;
            smem[i2] = a - b + c + d;
            smem[i3] = -a + b + c + d;
        }
        __syncthreads();
    }

    output[row * k + col] = static_cast<scalar_t>(smem[tid] * inv_norm);
}

}  // namespace

torch::Tensor hadamard_rotate_cuda(
    const torch::Tensor& input,
    int64_t rot_size) {
    if (!input.is_cuda()) {
        throw std::runtime_error("hadamard_rotate_cuda: input must be CUDA tensor");
    }
    if (input.scalar_type() != torch::kHalf && input.scalar_type() != torch::kBFloat16) {
        throw std::runtime_error("hadamard_rotate_cuda: input dtype must be fp16 or bf16");
    }

    auto is_power_of_four = [](int64_t v) -> bool {
        if (v < 4) {
            return false;
        }
        while ((v % 4) == 0) {
            v /= 4;
        }
        return v == 1;
    };
    if (rot_size > 1024 || !is_power_of_four(rot_size)) {
        throw std::runtime_error("hadamard_rotate_cuda: rot_size must be power-of-4 and <= 1024");
    }

    auto input_contig = input.contiguous();
    auto shape = input_contig.sizes().vec();
    if (shape.empty()) {
        throw std::runtime_error("hadamard_rotate_cuda: input rank must be >= 1");
    }
    int64_t k = shape.back();
    if (k <= 0) {
        throw std::runtime_error("hadamard_rotate_cuda: input last dim must be > 0");
    }

    int64_t batch = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        batch *= shape[i];
    }

    auto flat = input_contig.view({batch, k});
    int64_t num_blocks = (k + rot_size - 1) / rot_size;
    dim3 grid(batch, static_cast<unsigned int>(num_blocks));
    dim3 block(static_cast<unsigned int>(rot_size));
    size_t shared_bytes = static_cast<size_t>(rot_size) * sizeof(float);
    float inv_norm = 1.0f / std::sqrt(static_cast<float>(rot_size));
    bool full_blocks = (k % rot_size == 0);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty_like(flat);

    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, flat.scalar_type(), "rotated_linear_cuda", [&] {
        auto launch_static = [&]<int RS>() {
            dim3 static_block(static_cast<unsigned int>(RS));
            size_t static_shared = static_cast<size_t>(RS) * sizeof(float);
            if (full_blocks) {
                dim3 static_grid(batch, static_cast<unsigned int>(k / RS));
                hadamard_rotate_kernel_static<RS, true, scalar_t><<<static_grid, static_block, static_shared, stream>>>(
                    flat.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), batch, k, num_blocks, inv_norm);
                return;
            }

            hadamard_rotate_kernel_static<RS, false, scalar_t><<<grid, static_block, static_shared, stream>>>(
                flat.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), batch, k, num_blocks, inv_norm);
        };

        switch (rot_size) {
            case 256:
                launch_static.template operator()<256>();
                break;
            case 1024:
                launch_static.template operator()<1024>();
                break;
            case 64:
                launch_static.template operator()<64>();
                break;
            case 16:
                launch_static.template operator()<16>();
                break;
            case 4:
                launch_static.template operator()<4>();
                break;
            default:
                if (full_blocks) {
                    hadamard_rotate_kernel_full<scalar_t><<<grid, block, shared_bytes, stream>>>(
                        flat.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), batch, k, rot_size, inv_norm);
                    return;
                }

                hadamard_rotate_kernel_general<scalar_t><<<grid, block, shared_bytes, stream>>>(
                    flat.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), batch, k, rot_size, num_blocks, inv_norm);
                break;
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.view(shape);
}

torch::Tensor rotated_linear_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    c10::optional<torch::Tensor> bias_opt,
    int64_t rot_size) {
    auto rotated = hadamard_rotate_cuda(input, rot_size);
    return at::linear(rotated, weight, bias_opt);
}

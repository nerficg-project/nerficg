#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

/*
Implementation based on
1. https://github.com/m-schuetz/compute_rasterizer/blob/f2cbb658e6bf58407c385c75d21f3f615f11d5c9/tools/sort_points/Sort_Frugal/src/main.cpp#L79
2. https://gitlab.inria.fr/sibr/sibr_core/-/blob/gaussian_code_release_linux/src/projects/gaussianviewer/renderer/GaussianView.cpp?ref_type=heads#L90
*/

__constant__ float d_cube_size;
__constant__ float3 d_minimum_coordinates;

__device__ __forceinline__ uint64_t splitBy3(uint32_t a) {
	uint64_t x = a & 0x1fffff;
	x = (x | x << 32) & 0x1f00000000ffff;
	x = (x | x << 16) & 0x1f0000ff0000ff;
	x = (x | x << 8) & 0x100f00f00f00f00f;
	x = (x | x << 4) & 0x10c30c30c30c30c3;
	x = (x | x << 2) & 0x1249249249249249;
	return x;
}

__global__ void morton_encode_cu(
    const float3* __restrict__ positions,
    const float3* __restrict__ minimum_coordinates,
    const float*  __restrict__ cube_size,
    int64_t* __restrict__ morton_encoding,
    const uint32_t n_positions)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_positions) return;
    const float3 position = positions[idx];
    const float3 min = minimum_coordinates[0];
    const float cube_size_rcp = 1.0f / cube_size[0];
    const float normalized_x = __saturatef((position.x - min.x) * cube_size_rcp);
    const float normalized_y = __saturatef((position.y - min.y) * cube_size_rcp);
    const float normalized_z = __saturatef((position.z - min.z) * cube_size_rcp);
    constexpr float factor = 2097151.0f; // 2^21 - 1
    const uint32_t x = static_cast<uint32_t>(normalized_x * factor);
    const uint32_t y = static_cast<uint32_t>(normalized_y * factor);
    const uint32_t z = static_cast<uint32_t>(normalized_z * factor);
    const uint64_t morton_code = splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    morton_encoding[idx] = static_cast<int64_t>(morton_code); // most significant bit is zero by construction
}

torch::Tensor morton_encode(const torch::Tensor& positions) {
    TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
    TORCH_CHECK(positions.dtype() == torch::kFloat32, "positions must be float32");
    TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3, "positions must have shape (N, 3)");

    auto minmax = positions.aminmax(0);
    auto& minimum_coordinates = std::get<0>(minmax);
    auto& maximum_coordinates = std::get<1>(minmax);
    auto cube_size = (maximum_coordinates - minimum_coordinates).amax();

    const uint32_t n_positions = positions.size(0);
    auto morton_encoding = torch::empty({n_positions}, positions.options().dtype(torch::kLong));

    constexpr uint32_t block_size = 256;
    const uint32_t grid_size = (n_positions + block_size - 1) / block_size;
    auto stream = at::cuda::getCurrentCUDAStream();
    morton_encode_cu<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const float3*>(positions.data_ptr<float>()),
        reinterpret_cast<const float3*>(minimum_coordinates.data_ptr<float>()),
        cube_size.data_ptr<float>(),
        morton_encoding.data_ptr<int64_t>(),
        n_positions
    );

    return morton_encoding;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("morton_encode", &morton_encode);
}

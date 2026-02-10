#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>

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
    const float3* positions,
    int64_t* morton_encoding,
    const uint32_t n_positions)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_positions) return;
    const float3 position = positions[idx];
    const float normalized_x = __saturatef((position.x - d_minimum_coordinates.x) / d_cube_size);
    const float normalized_y = __saturatef((position.y - d_minimum_coordinates.y) / d_cube_size);
    const float normalized_z = __saturatef((position.z - d_minimum_coordinates.z) / d_cube_size);
    constexpr float factor = 2097151.0f; // 2^21 - 1
    const uint32_t x = static_cast<uint32_t>(normalized_x * factor);
    const uint32_t y = static_cast<uint32_t>(normalized_y * factor);
    const uint32_t z = static_cast<uint32_t>(normalized_z * factor);
    const uint64_t morton_code = splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    morton_encoding[idx] = static_cast<int64_t>(morton_code); // most significant bit is zero by construction
}


at::Tensor morton_encode(
    const at::Tensor& positions,
    const at::Tensor& minimum_coordinates,
    const at::Tensor& cube_size)
{
    cudaMemcpyToSymbol(d_cube_size, cube_size.contiguous().data_ptr<float>(), sizeof(float));
    cudaMemcpyToSymbol(d_minimum_coordinates, minimum_coordinates.contiguous().data_ptr<float>(), sizeof(float3));

    const uint32_t n_positions = positions.size(0);
    at::Tensor morton_encoding = torch::empty({n_positions}, positions.options().dtype(torch::kLong));

    constexpr uint32_t block_size = 256;
    const uint32_t grid_size = (n_positions + block_size - 1) / block_size;
    morton_encode_cu<<<grid_size, block_size>>>(
        reinterpret_cast<const float3*>(positions.data_ptr<float>()),
        morton_encoding.data_ptr<int64_t>(),
        n_positions
    );

    return morton_encoding;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("morton_encode_cuda", &morton_encode);
}

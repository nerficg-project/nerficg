from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ENABLE_NVCC_LINEINFO = False  # set to True for profiling kernels with Nsight Compute (overhead is minimal)

module_root = Path(__file__).parent.absolute()
extension_name = module_root.name
extension_root = module_root / extension_name

cxx_flags, nvcc_flags = [], []
if ENABLE_NVCC_LINEINFO:
    nvcc_flags.append('-lineinfo')

cuda_extension = CUDAExtension(
    name=f'{extension_name}._C',
    sources=[str(module_root / 'morton_encoding.cu')],
    extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags}
)

setup(
    name=extension_name,
    author='Florian Hahlbohm',
    ext_modules=[cuda_extension],
    description='CUDA-accelerated morton code computation for 3d points.',
    cmdclass={'build_ext': BuildExtension}
)

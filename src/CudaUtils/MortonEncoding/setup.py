import os
from glob import glob
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

__author__ = 'Florian Hahlbohm'
__description__ = 'CUDA-accelerated morton code computation for 3D points.'

ENABLE_NVCC_LINEINFO = False  # set to True for profiling kernels with Nsight Compute (overhead is minimal)

module_root = Path(__file__).parent.absolute()
extension_name = module_root.name
extension_root = module_root / extension_name

# set up compiler flags
cxx_flags = ['/std:c++17' if os.name == 'nt' else '-std=c++17']
nvcc_flags = ['-std=c++17']
if ENABLE_NVCC_LINEINFO:
    nvcc_flags.append('-lineinfo')

# define the CUDA extension
extension = CUDAExtension(
    name=f'{extension_name}._C',
    sources=glob(str(extension_root / '**' / '*.cu'), recursive=True),
    extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags}
)

# set up the package
setup(
    name=extension_name,
    author=__author__,
    packages=[extension_name],
    ext_modules=[extension],
    description=__description__,
    cmdclass={'build_ext': BuildExtension}
)

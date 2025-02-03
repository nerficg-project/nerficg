import glob
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = Path(__file__).parent.absolute()
include_dirs = [str(ROOT_DIR / 'csrc' / 'include')]
# "helper_math.h" is copied from https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h

sources = glob.glob(str(ROOT_DIR / 'csrc' / '*.cpp')) + glob.glob(str(ROOT_DIR / 'csrc' / '*.cu'))


setup(
    name='VolumeRenderingV2',
    version='2.0',
    author='kwea123',
    author_email='kwea123@gmail.com',
    description='cuda volume rendering library',
    long_description='cuda volume rendering library',
    ext_modules=[
        CUDAExtension(
            name='VolumeRenderingV2',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

# -- coding: utf-8 --

"""
Thirdparty/TinyCudaNN.py: Fast fused MLP and input encoding from NVLabs (https://github.com/NVlabs/tiny-cuda-nn).
"""

import Framework

__extension_name__ = 'tinycudann'
__install_command__ = [
    'pip', 'install',
    'git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch',
]

try:
    from tinycudann import * # noqa
except ImportError:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)

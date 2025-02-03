# -- coding: utf-8 --

"""
Thirdparty/TorchScatter.py: Torch Scatter lib.
"""

import Framework
import torch

__extension_name__ = 'torch-scatter'
__install_command__ = [
    'pip', 'install',
    'torch-scatter',
    '-f', f'https://data.pyg.org/whl/torch-{torch.__version__}.html',
]

try:
    from torch_scatter import segment_csr  # noqa
except ImportError:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)

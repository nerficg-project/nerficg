# -- coding: utf-8 --

from pathlib import Path

import Framework

__extension_name__ = 'DiffGaussianRasterization'
REPO_URL = 'https://github.com/graphdeco-inria/diff-gaussian-rasterization'
COMMIT_HASH = '59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d'
__install_command__ = ['pip', 'install', f'git+{REPO_URL}@{COMMIT_HASH}']

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # noqa
except ImportError:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)

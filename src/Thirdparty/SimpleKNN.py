"""
Thirdparty/SimpleKNN.py: A fast cuda implementation for knn distance computation from https://github.com/camenduru/simple-knn.
"""

import Framework

__extension_name__ = 'simple-knn'
REPO_URL = 'https://github.com/camenduru/simple-knn'
COMMIT_HASH = '60f461f4a56b7967e5d8045bf92f8c33f36976d0'
__install_command__ = [
    'pip', 'install',
    f'git+{REPO_URL}@{COMMIT_HASH}',
    '--no-build-isolation'
]

try:
    from simple_knn import _C  # noqa
    compute_mean_squared_knn_distances = _C.distCUDA2
except ImportError:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)

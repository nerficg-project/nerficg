"""
Thirdparty/Apex.py: NVIDIA Apex (https://github.com/NVIDIA/apex).
"""

import Framework

__extension_name__ = 'apex'
__install_command__ = [
    'pip', 'install',
    '-v', '--disable-pip-version-check', '--no-cache-dir', '--no-build-isolation',
    '--config-settings', '--build-option=--cpp_ext',
    '--config-settings', '--build-option=--cuda_ext',
    'git+https://github.com/NVIDIA/apex',
]

try:
    from apex.optimizers import FusedAdam # noqa
except ImportError:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)

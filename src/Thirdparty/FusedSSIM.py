"""
Thirdparty/FusedSSIM.py: fast cuda ssim implementation from https://github.com/rahul-goel/fused-ssim.
"""

import Framework

__extension_name__ = 'FusedSSIM'
__install_command__ = [
    'pip', 'install',
    'git+https://github.com/rahul-goel/fused-ssim/',
    '--no-build-isolation',
]

try:
    from fused_ssim import fused_ssim
except ImportError:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)

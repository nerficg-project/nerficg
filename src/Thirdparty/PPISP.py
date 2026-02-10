"""
Thirdparty/PPSIP.py: post-processing module for common photometric variations from https://github.com/nv-tlabs/ppisp.
"""

import Framework

__extension_name__ = 'PPISP'
__install_command__ = [
    'pip', 'install',
    'git+https://github.com/nv-tlabs/ppisp/',
    '--no-build-isolation',
]

try:
    from ppisp import PPISP, PPISPConfig
except ImportError:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)

"""Volume Rendering code adapted from pytorch InstantNGP reimplementation, adapted from kwea123 (https://github.com/kwea123/ngp_pl)"""

from pathlib import Path

import Framework

filepath = Path(__file__).resolve()
__extension_name__ = filepath.parent.stem
__install_command__ = [
    'pip', 'install',
    str(Path(__file__).parent),
    '--no-build-isolation',
]

try:
    from VolumeRenderingV2 import *  # noqa
    from .custom_functions import *  # noqa
except ImportError:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)

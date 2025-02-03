# -- coding: utf-8 --

from pathlib import Path

import Framework

__extension_name__ = 'simple-knn'
REPO_URL = 'https://github.com/camenduru/simple-knn'
COMMIT_HASH = '44f764299fa305faf6ec5ebd99939e0508331503'
__install_command__ = ['pip', 'install', f'git+{REPO_URL}@{COMMIT_HASH}']

try:
    from simple_knn._C import distCUDA2  # noqa
except ImportError:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)

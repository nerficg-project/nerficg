"""
Thirdparty/Transformers.py: Provides access to various state-of-the-art machine learning models via https://github.com/huggingface/transformers/.
Note: While the transformers library itself is licensed under Apache 2.0, individual models may be subject to stricter license terms.
"""

import Framework

__extension_name__ = 'transformers'
__install_command__ = [
    'pip', 'install',
    'transformers>=5', 'accelerate'
]

try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    from transformers import Sam3Model, Sam3Processor
except ImportError:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)

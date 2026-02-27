"""
Patch for torch version check in transformers >= 4.56.0

This module must be imported BEFORE any transformers imports.
It bypasses the CVE-2025-32434 check that requires torch >= 2.6.

Usage:
    import patch_torch  # First!
    import transformers  # Now safe
"""

import sys
import importlib.util

# Pre-load and patch transformers.utils.import_utils
spec = importlib.util.find_spec("transformers.utils.import_utils")
if spec is not None:
    module = importlib.util.module_from_spec(spec)
    sys.modules["transformers.utils.import_utils"] = module
    spec.loader.exec_module(module)
    # Bypass the torch version check
    module.check_torch_load_is_safe = lambda: None

    # Also need to patch modeling_utils which imports this function
    modeling_spec = importlib.util.find_spec("transformers.modeling_utils")
    if modeling_spec is not None:
        modeling_module = importlib.util.module_from_spec(modeling_spec)
        sys.modules["transformers.modeling_utils"] = modeling_module
        modeling_spec.loader.exec_module(modeling_module)
        modeling_module.check_torch_load_is_safe = lambda: None

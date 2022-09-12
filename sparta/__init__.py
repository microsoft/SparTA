# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

try:
    import torch
    import pycuda
except ImportError:
    torch = None
    pycuda = None
finally:
    __env_ready__ = torch is not None and pycuda is not None


from sparta import nn, testing

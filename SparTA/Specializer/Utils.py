# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ctypes
import logging


_logger = logging.Logger(__name__)

def cuda_detect():
    """
    Detect the cuda environment.
    """
    def ConvertSMVer2Cores(major, minor):
        # Returns the number of CUDA cores per multiprocessor for a given
        # Compute Capability version. There is no way to retrieve that via
        # the API, so it needs to be hard-coded.
        # See _ConvertSMVer2Cores in helper_cuda.h in NVIDIA's CUDA Samples.
        return {(1, 0): 8,    # Tesla
                (1, 1): 8,
                (1, 2): 8,
                (1, 3): 8,
                (2, 0): 32,   # Fermi
                (2, 1): 48,
                (3, 0): 192,  # Kepler
                (3, 2): 192,
                (3, 5): 192,
                (3, 7): 192,
                (5, 0): 128,  # Maxwell
                (5, 2): 128,
                (5, 3): 128,
                (6, 0): 64,   # Pascal
                (6, 1): 128,
                (6, 2): 128,
                (7, 0): 64,   # Volta
                (7, 2): 64,
                (7, 5): 64,   # Turing
                (8, 0): 64,   # Ampere
                (8, 6): 64,
                }.get((major, minor), 0)
    # Some constants taken from cuda.h
    CUDA_SUCCESS = 0
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError("could not load any of: " + ' '.join(libnames))

    nGpus = ctypes.c_int()
    name = b' ' * 100
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()

    result = ctypes.c_int()
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        _logger.warning("cuInit failed with error code %d: %s" , result, error_str.value.decode())
        return
    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        _logger.warning("cuDeviceGetCount failed with error code %d: %s", result, error_str.value.decode())
        return
    devices = []
    for i in range(nGpus.value):
        result = cuda.cuDeviceGet(ctypes.byref(device), i)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            _logger.warning("cuDeviceGet failed with error code %d: %s", result, error_str.value.decode())
            return
        device_name = None
        device_code = None
        if cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) == CUDA_SUCCESS:
            device_name = name.split(b'\0', 1)[0].decode()
        if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) == CUDA_SUCCESS:
            device_code = "%d%d" % (cc_major.value, cc_minor.value)
        devices.append((device_name, device_code))
    return devices

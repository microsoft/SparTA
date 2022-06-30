# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from socket import if_indextoname
import torch
import copy
import ctypes
import logging
import subprocess

_logger = logging.Logger(__name__)

def convert_bcsr(weight_m, weight_v, block_h=1, block_w=1):
    """
    convert the dense values to the block csr format according
    to the mask.
    """
    assert len(weight_m.size()) == 2, "Only support two dimension"
    weight_m = torch.abs(weight_m)
    size_h, size_w = weight_m.size()
    if size_h % block_h != 0 or size_w % block_w != 0:
        return None, None, None
    rows = []
    cols = []
    values = []
    for _i in range(size_h//block_h):
        rows.append(len(cols))
        for _j in range(size_w//block_w):
            i_start = _i * block_h
            i_end = (_i+1) * block_h
            j_start = _j * block_w
            j_end = (_j+1) * block_w
            if torch.sum(weight_m[i_start:i_end, j_start:j_end]) > 0:
                cols.append(_j)
                values.extend(weight_v[i_start:i_end,j_start:j_end].flatten().tolist())
    rows.append(len(cols))
    t_rows = torch.tensor(rows).to(torch.int32)
    t_cols = torch.tensor(cols).to(torch.int32)
    t_values = torch.tensor(values)
    return t_rows, t_cols, t_values

def verify_bcsr(ori_mask, value, csr_row, csr_col, csr_val, block_h, block_w):
    """
    Verify if the converted bcsr format is right.
    """
    assert isinstance(ori_mask, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    assert isinstance(csr_row, torch.Tensor)
    assert isinstance(csr_col, torch.Tensor)
    assert isinstance(csr_val, torch.Tensor)

    value = value.cpu()
    csr_row = csr_row.cpu()
    csr_col = csr_col.cpu()
    csr_val = csr_val.cpu()
    mask = copy.deepcopy(ori_mask.cpu())
    n_row = mask.size(0) // block_h
    csr_val = csr_val.flatten()
    for rid in range(n_row):
        _start = csr_row[rid]
        _end = csr_row[rid+1]
        for _pos in range(_start, _end):
            cid = csr_col[_pos]
            _r_start = rid * block_h
            _c_start = cid * block_w
            for i in range(block_h):
                for j in range(block_w):
                    if mask[_r_start+i][_c_start+j] > 0:
                        assert(torch.abs(value[_r_start+i][_c_start+j]-csr_val[_pos*block_h*block_w+i*block_w+j])<1e-6)
                        mask[_r_start+i][_c_start+j] = 0
    if torch.sum(mask) > 0:
        return False
    return True

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

def call_shell(cmd: str, logger: logging.Logger = _logger):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    for line in filter(lambda x: x, stdout.decode("utf-8").split('\n')):
        logger.info(line)
    for line in filter(lambda x: x, stderr.decode("utf-8").split('\n')):
        logger.error(line)

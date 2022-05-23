# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import copy

def convert_bcsr(weight_m, weight_v, block_h=1, block_w=1):
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
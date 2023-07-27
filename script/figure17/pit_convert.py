# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import torch
import random
import csv
from sparta.opset import *
from sparta.common.utils import convert_bcsr, verify_bcsr

def convert_to_full_mask(block_layout, block_size):
    full_mask = block_layout.repeat_interleave(block_size[0], dim=0)
    full_mask = full_mask.repeat_interleave(block_size[1], dim=1)
    return full_mask


def test(h, w, block_h, block_w, sparsity):

    device = torch.device('cuda')
    block_weight = torch.rand(h//block_h, w//block_w).to(device)
    block_mask = (block_weight > sparsity).to(torch.int32)
    mask = convert_to_full_mask(block_mask, (block_h, block_w))
    print('real sparsity: ', 1 - torch.sum(block_mask)/block_mask.numel())

    dense_value = torch.rand(h, w).to(device)
    dense_value *= mask
    converter = BcsrConverter()
    RUNTIME = 1000
    torch.cuda.synchronize()    
    t_start = time.time()
    for i in range(RUNTIME):
        row, col, row_pos, value = converter(mask, dense_value, block_h, block_w)
    torch.cuda.synchronize()
    t_end = time.time()
    time_re = (t_end-t_start)*1000/RUNTIME
    print(f"H:{h} W:{w} Block_h:{block_h} Block_w:{block_w} Sparsity:{sparsity} Time: ", time_re)
    return time_re

def test_correctness():
    h, w = 4096, 4096
    sparsity=0.5
    weight = torch.rand(h,w).cuda()
    mask = (weight>sparsity).to(torch.int32)
    converter = BcsrConverter()
    row, col, _, value = converter(mask, weight, 1, 1)
    import ipdb; ipdb.set_trace()
    assert verify_bcsr(mask, weight, row, col, value, 1, 1)

if __name__ == '__main__':
    # test_correctness()
    with open('convert.csv', 'w') as f:
        f.write('h,w,block_h,block_w,sparsity,t_avg\n')
        writer = csv.writer(f, delimiter=',')
        for h, w in [(4096, 4096)]:
            for block_h, block_w in [(1,1), (16,16), (32,32)]:
            # for block_h, block_w in [(1,1)]:
                for sparsity in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
                    t_avg = test(h, w, block_h, block_w, sparsity)
                    writer.writerow(str(c) for c in [h, w, block_h, block_w, sparsity, t_avg])
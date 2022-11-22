# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Callable, List

import torch


def profile(
    func: Callable, inputs: List, target_outputs: Optional[List] = None,
    num_warmups: int = 10, num_iters: int = 10, cuda: bool = False
) -> float:
    '''Test latency of a CUDA function.

    Args:
        func (Callable): A function that runs on CUDA GPUs.
        inputs (list): Input variables (tensors).
        target_outputs (list): Target output variables (tensors).
        num_warmups (int): Number of warm-up iterations.
        num_iters (int): Number of test iterations.
        cuda (bool): Whether to return kernel-level CUDA time.

    Returns:
        float: latency in milliseconds.
    '''
    check(func, inputs, target_outputs)
    try:
        if target_outputs is not None:
            check(func, inputs, target_outputs)
        torch.cuda.synchronize()
        for _ in range(num_warmups):
            func(*inputs)
        torch.cuda.synchronize()
        if cuda:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
                for _ in range(num_iters):
                    func(*inputs)
            latency = 0
            for event in p.key_averages():
                if event.key != 'cudaDeviceSynchronize':
                    latency += event.cuda_time * event.count
            latency /= num_iters * 1000
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(num_iters):
                func(*inputs)
            end.record()
            torch.cuda.synchronize()
            latency = start.elapsed_time(end) / num_iters
    except:
        latency = float('inf')
    return latency


def check(func: Callable, inputs: List, target_outputs: List):
    '''Test correctness of a CUDA function.

    Args:
        func (Callable): A function that runs on CUDA GPUs.
        inputs (list): Input variables (tensors).
        target_outputs (list): Target output variables (tensors).
    '''
    outputs = func(*inputs)
    if len(target_outputs) == 1:
        outputs = [outputs]
    assert len(outputs) == len(target_outputs), f'expected {len(target_outputs)} outputs, got {len(outputs)}'
    for output, target_output in zip(outputs, target_outputs):
        torch.testing.assert_close(output, target_output, atol=1e-4, rtol=1e-4)

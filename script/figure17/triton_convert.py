import torch
import sparta
import time
import argparse
from sparta.opset.triton_dynamic_sparse_linear import TritonDynamicLinear

    
def convert_to_full_mask(block_layout, block_size):
    full_mask = block_layout.repeat_interleave(block_size[0], dim=0)
    full_mask = full_mask.repeat_interleave(block_size[1], dim=1)
    return full_mask

parser = argparse.ArgumentParser()
parser.add_argument('--block', type=int, required=True, help='Block size')
args = parser.parse_args()
if __name__ == '__main__':
    B = 8
    S = 512
    K = 4096
    N = 4096
    block_h = args.block
    block_w = args.block
    
    for sparsity_ratio in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        block_wise_weight = torch.rand(N//block_h, K//block_w, dtype=torch.float32).cuda()
        block_mask = (block_wise_weight > sparsity_ratio).to(torch.int32)
        full_mask = convert_to_full_mask(block_mask, (block_h, block_w))
        print("Sparsity ratio:", torch.sum(block_mask)/block_mask.numel())
        data =  torch.rand(B, S, K).cuda()
        # data =  torch.ones(B, S, K).cuda()
        ori_linear = torch.nn.Linear(K, N).cuda()
        ori_linear.weight.data *= full_mask.data
        
        # ori_linear.weight.data[:] = 1
        ori_linear.bias.data[:] = 0
        t_linear = TritonDynamicLinear(ori_linear, block_h, block_w, profile=True)
        RUNTIME = 100
        for i in range(RUNTIME):
            t_linear(data, full_mask)
        latency = sum(t_linear.convert_overhead)/len(t_linear.convert_overhead)
        print('latency :', latency)
        with open(f'log/{args.block}_{sparsity_ratio}.log', 'w') as f:
            f.write('latency: {}'.format(latency))
            
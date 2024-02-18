import torch
import sparta

batch_size, in_features, out_features = 1024, 1024, 1024
sparsity = 0.9
granularity = (8, 8)

# prepare data
x = torch.rand((batch_size, in_features), device='cuda')
weight = torch.rand((out_features, in_features), device='cuda')
bias = torch.rand((out_features, ), device='cuda')

# generate and apply weight mask
mask = sparta.testing.block_mask(weight.shape, granularity, sparsity, device='cuda')
weight = torch.mul(weight, mask)

# create a dense operator
dense_linear = torch.nn.Linear(in_features, out_features, device='cuda')
dense_linear.load_state_dict({'weight': weight, 'bias': bias})

# create a sparse operator
sparse_linear = sparta.nn.SparseLinear(dense_linear)
sparse_linear.set_mask(mask)

# tune the sparse operator
best_config = sparta.nn.tune(sparse_linear, sample_inputs=[x], max_trials=10, algo='rand')

# check if the sparse operator runs correctly
torch.testing.assert_close(sparse_linear(x), dense_linear(x))
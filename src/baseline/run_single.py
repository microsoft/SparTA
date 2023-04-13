import os
import sys
shape_list=[ (4096,5120,20480)]
sparsity_ratio=[0.5, 0.75, 0.90625]
for shape_id in range(len(shape_list)):
    for sparsity in sparsity_ratio:    
        m, k,  n = shape_list[shape_id]
        print(f"{m} {k} {n} {sparsity}")
        os.system(f'./cublas {sparsity} {n} {k} {m} > log/cublas_M{shape_id}_{sparsity:.2f}.log')
        os.system(f'./cusparse {sparsity} {m} {k} {n} > log/cusparse_M{shape_id}_{sparsity:.2f}.log')
        os.system(f'./cusparse_block_ell {sparsity} {n} {k} {m} > log/cusparseblockELL_M{shape_id}_{sparsity:.2f}.log')
        os.system(f'./sputnik {sparsity} {n} {k} {m} > log/sputnik_M{shape_id}_{sparsity:.2f}.log')
    os.system(f'./cusparselt 0 {n} {k} {m} > log/cusparselt_M{shape_id}_0.5.log')
    
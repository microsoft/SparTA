import os
import sys
shape_list=[ (4096,5120,20480)]
sparsity_ratio=[0.5, 0.75, 0.90625]
for shape_id in range(len(shape_list)):
    for sparsity in sparsity_ratio:    
        m, k,  n = shape_list[shape_id]
        print(f"{m} {k} {n} {sparsity}")
        os.system(f'./cublas {sparsity} {m} {k} {n} > log/cublas_M{shape_id}_{sparsity:.2f}.log')
        os.system(f'./cusparse {sparsity} {m} {k} {n} > log/cusparse_M{shape_id}_{sparsity:.2f}.log')
        os.system(f'./cusparse_block_ell {sparsity} {m} {k} {n} > log/cusparseblockELL_M{shape_id}_{sparsity:.2f}.log')
        os.system(f'./sputnik {sparsity} {m} {k} {n} > log/sputnik_M{shape_id}_{sparsity:.2f}.log')
    os.system(f'./cusparselt 0 {m} {k} {n} > log/cusparselt_M{shape_id}_0.5.log')
    
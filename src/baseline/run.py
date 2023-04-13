import os
import sys
shape_list=[ (1,1024,1024), (1,2048,2048), (1,4096,4096), (1,8192,8192), (1,1024,4096), (1,4096,1024), (1,5120,20480), (1,20480,5120), (256,1024,1024), (1024,1024,1024), (4096,1024,1024), (256,2048,2048), (1024,2048,2048), (4096,2048,2048), (256,4096,4096), (1024,4096,4096), (4096,4096,4096), (256,8192,8192), (1024,8192,8192), (4096,8192,8192), (256,1024,4096), (1024,1024,4096), (4096,1024,4096), (256,4096,1024), (1024,4096,1024), (4096,4096,1024), (256,5120,20480), (1024,5120,20480), (4096,5120,20480), (256,20480,5120), (1024,20480,5120), (4096,20480,5120) ]
sparsity_ratio=[0.5, 0.75, 0.90625]
for shape_id in range(1, len(shape_list) + 1):
    m, k, n = shape_list[shape_id]
    iterations = 10000 if m == 1 else 100
    # iterations = 1
    for sparsity in sparsity_ratio:    
        print(f"{m} {k} {n} {sparsity}")
        os.system(f'./cublas {sparsity} {n} {k} {m} {iterations} > log/cublas_M{shape_id}_{sparsity:.2f}.log')
        os.system(f'./cusparse {sparsity} {m} {k} {n} {iterations} > log/cusparse_M{shape_id}_{sparsity:.2f}.log')
        os.system(f'./cusparse_block_ell {sparsity} {n} {k} {m} {iterations} > log/cusparseblockELL_M{shape_id}_{sparsity:.2f}.log')
        os.system(f'./sputnik {sparsity} {n} {k} {m} {iterations} > log/sputnik_M{shape_id}_{sparsity:.2f}.log')
    os.system(f'./cusparselt 0 {n} {k} {m} {iterations} > log/cusparselt_M{shape_id}_0.5.log')
    os.system(f'./cusparselt_int8 {n} {k} {m} {iterations} > log/cusparselt_int8_M{shape_id}_0.5.log')

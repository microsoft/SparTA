import os
import sys
shape_list=[ (1,1024,1024), (1,2048,2048), (1,4096,4096), (1,8192,8192), (1,1024,4096), (1,4096,1024), (1,5120,20480), (1,20480,5120), (256,1024,1024), (1024,1024,1024), (4096,1024,1024), (256,2048,2048), (1024,2048,2048), (4096,2048,2048), (256,4096,4096), (1024,4096,4096), (4096,4096,4096), (256,8192,8192), (1024,8192,8192), (4096,8192,8192), (256,1024,4096), (1024,1024,4096), (4096,1024,4096), (256,4096,1024), (1024,4096,1024), (4096,4096,1024), (256,5120,20480), (1024,5120,20480), (4096,5120,20480), (256,20480,5120), (1024,20480,5120), (4096,20480,5120) ]
sparsity_ratio=[0.5, 0.6, 0.7, 0.8, 0.9]
for shape_id in range(len(shape_list)):
    for sparsity in sparsity_ratio:    
        m, k,  n = shape_list[shape_id]
        print(f"{m} {k} {n} {sparsity}")
        os.system(f'./cublas {sparsity} {m} {k} {n} > log/cublas_M{shape_id}_{sparsity:.2f}.log')
        os.system(f'./cusparse {sparsity} {m} {k} {n} > log/cusparse_M{shape_id}_{sparsity:.2f}.log')
        os.system(f'./sputnik {sparsity} {m} {k} {n} > log/sputnik_M{shape_id}_{sparsity:.2f}.log')
    os.system(f'./cusparselt 0 {m} {k} {n} > log/cusparselt_M{shape_id}_0.5.log')
    
nvcc -lcublas -o cublas cublas.cu
nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75  -lcusparse -o cusparse cusparse.cu
nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75   -O3  stile_finegrained.cu -o stile_finegrained
nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75   -O3  stile.cu -o stile
nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75   -O3  2.hgemm-tc.cu -o hgemm
nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75   -O3  block_sparse_fp16.cu -o block_sparse_fp16
nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75   -O3  block_sparse_fp16_k.cu -o block_sparse_fp16_k
nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75   -O3  test_condense_k.cu -o test_condense_k
nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75   -O3  test_condense_m.cu -o test_condense_m
SPUTNIK_ROOT=/root/sputnik
nvcc -forward-unknown-to-host-compiler  -I/usr/local/cuda/include -I${SPUTNIK_ROOT} -I${SPUTNIK_ROOT}/third_party/abseil-cpp -L/usr/local/cuda/lib64  -L${SPUTNIK_ROOT}/build/sputnik  -lcusparse -lcudart -lspmm  --generate-code=arch=compute_70,code=sm_70 -std=c++14  sputnik.cu -o sputnik

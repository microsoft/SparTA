source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact
nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75   -O3  block_sparse_fp16.cu -o block_sparse_fp16
nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75   -O3  block_sparse_fp16_k.cu -o block_sparse_fp16_k
mkdir -p log
for sparsity in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99
do
    ./block_sparse_fp16 $sparsity > log/ori_${sparsity}.log
    ./block_sparse_fp16_k $sparsity > log/pit_${sparsity}.log
done

python plot.py

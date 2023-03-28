# source ~/anaconda/etc/profile.d/conda.sh
# conda activate artifact

# nvcc openai_blocksparse.cu -o openai_blocksparse
# nvcc -lcublas -o cublas cublas.cu
# nvcc -gencode arch=compute_80,code=sm_80 -lcusparse -o cusparse cusparse.cu
#sparsity_ratio=(0.5 0.6 0.7 0.8 0.9)
#sparsity_ratio=(0.5 0.75 0.9)
sparsity_ratio=(0.5)
#M=(1 16 256 1024 4096)
#KN=(1024 2048 4096 8192)
M=(1 16 256 1024 4096)
K=(1024 2048 4096 5120 8192 20480)
N=(1024 2048 4096 5120 8192 20480)
mkdir -p log
for sparsity in ${sparsity_ratio[@]}
do
    for m in ${M[@]}
    do
        for k in ${K[@]}
	do
	    for n in ${N[@]}
	    do
                echo $m $k $n $sparsity
                python ./src/nmsparse/run_balance_align_reg_block.py --sparsity_ratio $sparsity --M $m --K $k --N $n > log/nmsparse_run_balance_align_reg_block_${m}_${k}_${n}_${sparsity}.log
                python ./src/nmsparse/run_balance_align_reg.py --sparsity_ratio $sparsity --M $m --K $k --N $n > log/run_balance_align_reg_${m}_${k}_${n}_${sparsity}.log
                python ./src/nmsparse/run_balance_align.py --sparsity_ratio $sparsity --M $m --K $k --N $n > log/run_balance_align_${m}_${k}_${n}_${sparsity}.log
                python ./src/nmsparse/run_mv_one_kernel_block_batch.py --sparsity_ratio $sparsity --M $m --K $k --N $n > log/run_mv_one_kernel_block_batch_${m}_${k}_${n}_${sparsity}.log
                python ./src/nmsparse/run_mv_one_kerrun_balance_align_sharednel_block.py --sparsity_ratio $sparsity --M $m --K $k --N $n > log/run_balance_align_shared_${m}_${k}_${n}_${sparsity}.log
                python ./src/nmsparse/run_mv_one_kerrun_balance_align_sharednel_block_int8.py --sparsity_ratio $sparsity --M $m --K $k --N $n > log/run_balance_align_shared_int8_${m}_${k}_${n}_${sparsity}.log
            done
       	done
    done
done

# source ~/anaconda/etc/profile.d/conda.sh
# conda activate artifact

# SPUTNIK_ROOT=/root/sputnik
# CUSPARSELT_ROOT=/root/libcusparse_lt
# nvcc -forward-unknown-to-host-compiler  -I/usr/local/cuda/include -I${SPUTNIK_ROOT} -I${CUSPARSELT_ROOT}/include -I${SPUTNIK_ROOT}/third_party/abseil-cpp -L/usr/local/cuda/lib64  -L${SPUTNIK_ROOT}/build/sputnik -L${CUSPARSELT_ROOT}/lib64  -lcusparse -lcudart -lcusparseLt -lspmm  --generate-code=arch=compute_80,code=sm_80 -std=c++14  sparta.cu -o sparta
# nvcc -forward-unknown-to-host-compiler  -I/usr/local/cuda/include -I${SPUTNIK_ROOT} -I${CUSPARSELT_ROOT}/include -I${SPUTNIK_ROOT}/third_party/abseil-cpp -L/usr/local/cuda/lib64  -L${SPUTNIK_ROOT}/build/sputnik -L${CUSPARSELT_ROOT}/lib64  -lcusparse -lcudart -lcusparseLt -lspmm  --generate-code=arch=compute_80,code=sm_80 -std=c++14  cusparselt.cu -o cusparselt
# nvcc -forward-unknown-to-host-compiler  -I/usr/local/cuda/include -I${SPUTNIK_ROOT} -I${CUSPARSELT_ROOT}/include -I${SPUTNIK_ROOT}/third_party/abseil-cpp -L/usr/local/cuda/lib64  -L${SPUTNIK_ROOT}/build/sputnik -L${CUSPARSELT_ROOT}/lib64  -lcusparse -lcudart -lcusparseLt -lspmm  --generate-code=arch=compute_80,code=sm_80 -std=c++14  cusparselt_int8.cu -o cusparselt_int8
# nvcc -forward-unknown-to-host-compiler  -I/usr/local/cuda/include -I${SPUTNIK_ROOT} -I${CUSPARSELT_ROOT}/include -I${SPUTNIK_ROOT}/third_party/abseil-cpp -L/usr/local/cuda/lib64  -L${SPUTNIK_ROOT}/build/sputnik -L${CUSPARSELT_ROOT}/lib64  -lcusparse -lcudart -lcusparseLt -lspmm  --generate-code=arch=compute_80,code=sm_80 -std=c++14  sputnik.cu -o sputnik
# nvcc -forward-unknown-to-host-compiler  -I/usr/local/cuda/include -I${SPUTNIK_ROOT} -I${CUSPARSELT_ROOT}/include -I${SPUTNIK_ROOT}/third_party/abseil-cpp -L/usr/local/cuda/lib64  -L${SPUTNIK_ROOT}/build/sputnik -L${CUSPARSELT_ROOT}/lib64  -lcusparse -lcudart -lcusparseLt -lspmm  --generate-code=arch=compute_80,code=sm_80 -std=c++14  cusparse_block_ell.cu -o cusparse_block_ell

nvcc openai_blocksparse.cu -o openai_blocksparse
nvcc -lcublas -o cublas cublas.cu
nvcc -gencode arch=compute_80,code=sm_80 -lcusparse -o cusparse cusparse.cu
sparsity_ratio=(0.5 0.6 0.7 0.8 0.9)
#sparsity_ratio=(0.5 0.75 0.9)
# sparsity_ratio=(0.5)
#M=(1 16 256 1024 4096)
#KN=(1024 2048 4096 8192)
M=(1 16 256 1024 4096)
K=(1024 2048 4096 5120 8192)
# K=(1024 2048 4096 5120 8192 20480)
# N=(1024 2048 4096 5120 8192 20480)
N=(1024 2048 4096 5120 8192)
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
        		./cublas $sparsity $m $k $n > log/cublas_${m}_${k}_${n}_${sparsity}.log
                ./sputnik $sparsity  $m $k $n > log/sputnik_${m}_${k}_${n}_${sparsity}.log
                ./cusparse $sparsity  $m $k $n > log/cusparse_${m}_${k}_${n}_${sparsity}.log
            done
       	done
    done
done


for m in ${M[@]}
do
    for k in ${K[@]}
do
        for n in ${N[@]}
        do
            echo $m $k $n
            ./cusparselt 0 $m $k $n > log/cusparselt_${m}_${k}_${n}_0.5.log
        done
    done
done

#mkdir -p log
#for sparsity in ${sparsity_ratio[@]}
#do
#    echo $sparsity
#    ./cublas $sparsity > log/cublas_${sparsity}.log
#    ./openai_blocksparse $sparsity > log/openai_${sparsity}.log
#    ./sputnik $sparsity > log/sputnik_${sparsity}.log
#    ./sparta $sparsity > log/sparta_${sparsity}.log
#done
#python draw.py
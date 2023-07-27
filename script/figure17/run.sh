source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact

nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75  -lcusparse -o cusparse_convert cusparse_convert.cu

mkdir -p log
H=4096
W=4096

for sparsity in 0.5 0.9 0.95 0.99
do
    echo Sparsity:$sparsity
    ./cusparse_convert $sparsity $H $W > log/1_${sparsity}.log
done
python triton_convert.py --block 16
python triton_convert.py --block 32
python pit_convert.py

python plot.py

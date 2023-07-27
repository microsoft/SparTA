# Activate the conda environment
source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact

pip install git+https://github.com/iofu728/transformers.git@v4.20.0_PIT

SPARSE=(0.0209 0.0417 0.0625 0.0834 0.1042 0.1459 0.2084 0.2917 0.3959 0.5)

for sparse in ${SPARSE[@]}; do
    bash run_pytorch.sh ${sparse} 64 PyTorch pytorch
    bash run_pytorch.sh ${sparse} 64 PyTorch-S pytorch
    bash run_pytorch.sh ${sparse} 64 PIT pit
    bash run_pytorch.sh ${sparse} 1 PyTorch pytorch
    bash run_pytorch.sh ${sparse} 1 PyTorch-S pytorch
    bash run_pytorch.sh ${sparse} 1 PIT_32x1 pit
done

python plot.py
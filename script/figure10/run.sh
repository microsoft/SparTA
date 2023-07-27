
# Activate the conda environment
source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact

pip install git+https://github.com/iofu728/transformers.git@v4.20.0_PIT

azcopy copy "https://australiav100data.blob.core.windows.net/hjiang/Experiments/DynamicSparse/data.zip?sv=2021-04-10&st=2023-07-25T06%3A39%3A46Z&se=2024-07-26T06%3A39%3A00Z&sr=b&sp=r&sig=L2vyu71YJMTvnSryNHKgJHGcO1IP9O8a3yyS6wVA05g%3D" ./ --recursive
unzip data.zip

DATA_DIRS=(data/glue data/LongDocument)

for data_dir in ${DATA_DIRS[@]}; do
    CUDA_VISIBLE_DEVICES=0 python run_pytorch.py --data_dir ${data_dir}
    CUDA_VISIBLE_DEVICES=0 python run_pytorch_s.py --data_dir ${data_dir}
    CUDA_VISIBLE_DEVICES=0 python run_deepspeed.py --data_dir ${data_dir}
    CUDA_VISIBLE_DEVICES=0 python run_turbo.py --data_dir ${data_dir}
    CUDA_VISIBLE_DEVICES=0 python run_pit.py --data_dir ${data_dir}
done

python plot.py


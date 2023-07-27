
# Activate the conda environment
source ~/anaconda/etc/profile.d/conda.sh
conda activate longformer

# Download model weights
wget https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-base-4096.tar.gz
wget https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-large-4096.tar.gz
tar -zxvf longformer-base-4096.tar.gz
tar -zxvf longformer-large-4096.tar.gz

azcopy copy "https://australiav100data.blob.core.windows.net/hjiang/Experiments/DynamicSparse/data.zip?sv=2021-04-10&st=2023-07-25T06%3A39%3A46Z&se=2024-07-26T06%3A39%3A00Z&sr=b&sp=r&sig=L2vyu71YJMTvnSryNHKgJHGcO1IP9O8a3yyS6wVA05g%3D" ./ --recursive
unzip data.zip
cp data/*.pkl ./

MODELS=(longformer-base-4096/ longformer-large-4096/)
MAX_TOKENS=(2048 4096)

for model in ${MODELS[@]}; do
    for tokens in ${MAX_TOKENS[@]}; do
        python run_pytorch.py --model_name ${model} --max_seq_length ${tokens}
        python run_pytorch_s.py --model_name ${model} --max_seq_length ${tokens}
        python run_longformer_s.py --model_name ${model} --max_seq_length ${tokens}
        python run_deepspeed.py --model_name ${model} --max_seq_length ${tokens}
        python run_pit.py --model_name ${model} --max_seq_length ${tokens}
    done
done

python plot.py

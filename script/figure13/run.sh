# Activate the conda environment
source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact

pip install git+https://github.com/iofu728/transformers.git@v4.26.0_PIT

MODELS=(facebook/opt-125m facebook/opt-350m)

for model in ${MODELS[@]}; do
    python run_pit.py \
        --model_name_or_path ${model} \
        --dataset_name tatsu-lab/alpaca \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --do_train \
        --output_dir /tmp/test-clm \
        --overwrite_output_dir \
        --max_steps 100

    python run_pytorch.py \
        --model_name_or_path ${model} \
        --dataset_name tatsu-lab/alpaca \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --do_train \
        --output_dir /tmp/test-clm \
        --overwrite_output_dir \
        --max_steps 100

    python run_pytorch_s.py \
        --model_name_or_path ${model} \
        --dataset_name tatsu-lab/alpaca \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --do_train \
        --output_dir /tmp/test-clm \
        --overwrite_output_dir \
        --max_steps 100

    python run_deepspeed.py \
        --model_name_or_path ${model} \
        --dataset_name tatsu-lab/alpaca \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --do_train \
        --output_dir /tmp/test-clm \
        --overwrite_output_dir \
        --max_steps 100
done

python run_pit.py \
    --model_name_or_path facebook/opt-1.3b \
    --dataset_name tatsu-lab/alpaca \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --output_dir /tmp/test-clm \
    --overwrite_output_dir \
    --max_steps 100

python run_pytorch.py \
    --model_name_or_path facebook/opt-1.3b \
    --dataset_name tatsu-lab/alpaca \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --output_dir /tmp/test-clm \
    --overwrite_output_dir \
    --max_steps 100

python run_pytorch_s.py \
    --model_name_or_path facebook/opt-1.3b \
    --dataset_name tatsu-lab/alpaca \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --output_dir /tmp/test-clm \
    --overwrite_output_dir \
    --max_steps 100

python run_deepspeed.py \
    --model_name_or_path facebook/opt-1.3b \
    --dataset_name tatsu-lab/alpaca \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --output_dir /tmp/test-clm \
    --overwrite_output_dir \
    --max_steps 100

python plot.py
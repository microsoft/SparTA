# Activate the conda environment
source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact

pip install git+https://github.com/iofu728/transformers.git@v4.25.1_PIT triton==2.0.0.dev20221030

BATCHS=(8 32)
USE_FP16S=(True False)
EXPERT_NUMBERS=(64 128 256)

for batch in ${BATCHS[@]}; do
    for use_fp16 in ${USE_FP16S[@]}; do
        for exp_num in ${EXPERT_NUMBERS[@]}; do
            python run_pytorch.py --expert_number ${exp_num} --use_fp16 ${use_fp16} --batch_size ${batch}
            python run_pytorch_s.py --expert_number ${exp_num} --use_fp16 ${use_fp16} --batch_size ${batch}
            python run_deepspeed.py --expert_number ${exp_num} --use_fp16 ${use_fp16} --batch_size ${batch}
            python run_tutel.py --expert_number ${exp_num} --use_fp16 ${use_fp16} --batch_size ${batch}
            python run_megablocks.py --expert_number ${exp_num} --use_fp16 ${use_fp16} --batch_size ${batch}
            python run_pit.py --expert_number ${exp_num} --use_fp16 ${use_fp16} --batch_size ${batch}
            python run_pit_wo_moe.py --expert_number ${exp_num} --use_fp16 ${use_fp16} --batch_size ${batch}
        done
    done
done

python plot.py


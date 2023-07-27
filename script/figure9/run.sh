
# Activate the conda environment
source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact

# pip install -U datasets sentencepiece
pip install git+https://github.com/iofu728/transformers.git@v4.26.0_PIT

MODELS=(facebook/opt-13b facebook/opt-30b)

for model in ${MODELS[@]}; do
    python run_pytorch.py --name ${model}
    python run_pytorch_s.py --name ${model}
    python run_deepspeed.py --name ${model}
    python run_pit.py --name ${model}
    python run_pit_wo_activation.py --name ${model}
done

python plot.py


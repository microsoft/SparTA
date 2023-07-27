# Activate the conda environment
source ~/anaconda/etc/profile.d/conda.sh
conda activate longformer

# Training Data
azcopy copy "https://australiav100data.blob.core.windows.net/hjiang/Experiments/DynamicSparse/Museformer_data.zip?sv=2021-10-04&st=2023-07-25T12%3A50%3A06Z&se=2024-07-26T12%3A50%3A00Z&sr=b&sp=r&sig=jXmilWMp1mjSGQrBzaMizXpglmqvqfREIGmHbr4xQZc%3D" ./ --recursive
unzip Museformer_data.zip
cp -r Museformer_data/* ./

LENGTHS=(1024 4096 7168 15360 19456 23552 31744)

bash mf-lmd6remi-f1_pytorch.sh 1024

for lens in ${LENGTHS[@]}; do
    bash mf-lmd6remi-f1_pytorch.sh ${lens}
    bash mf-lmd6remi-f1_pytorch_s.sh ${lens}
    bash mf-lmd6remi-f1_deepspeed.sh ${lens}
    bash mf-lmd6remi-f1_pit.sh ${lens}
done

python plot.py

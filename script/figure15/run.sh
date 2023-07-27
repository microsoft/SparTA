source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact
# build the micro benchmark
bash build.sh

# run the micro benchmark for 32x1
mkdir -p log
for sparsity in 0.5 0.9 0.95 0.99
do
    echo Sparsity:$sparsity Block 32x1 
    ./sputnik $sparsity 4096 4096 4096 32 1 > log/sputnik_${sparsity}_32_1.log
    ./cusparse $sparsity 4096 4096 4096 32 1 > log/cusparse_${sparsity}_32_1.log
    python test_openai_bmm.py $sparsity 4096 4096 4096 32 1 > log/openai_${sparsity}_32_1.log
done

for sparsity in 0.5 0.9 0.95 0.99
do
    echo Sparsity:$sparsity Block 1x64 
    ./sputnik $sparsity 4096 4096 4096 1 64 > log/sputnik_${sparsity}_1_64.log
    ./cusparse $sparsity 4096 4096 4096 1 64 > log/cusparse_${sparsity}_1_64.log
    python test_openai_bmm.py $sparsity 4096 4096 4096 1 64 > log/openai_${sparsity}_1_64.log
done

for sparsity in 0.5 0.9 0.95 0.99
do
    echo Sparsity:$sparsity Block 32x64
    ./sputnik $sparsity 4096 4096 4096 32 64 > log/sputnik_${sparsity}_32_64.log
    ./cusparse $sparsity 4096 4096 4096 32 64 > log/cusparse_${sparsity}_32_64.log
    python test_openai_bmm.py $sparsity 4096 4096 4096 32 64 > log/openai_${sparsity}_32_64.log
done
python test_pit_bmm.py

python plot.py

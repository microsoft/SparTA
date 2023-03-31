# the code is placed in /root/nnfusion
# take the align-n = 4 and N:M = 3:32 as an example
conda activate artifact
align=4
remain=3
PWD=`pwd`
echo $PWD
pushd /root/nnfusion/Exp_Hardware
# generate the checkpoint under corrsponding sparsity pattern first 
python bert_large_balance_ck.py --align $align --remain $remain
bash run.sh $remain $align

pushd nnfusion_cfg_test/nnfusion_rt/cuda_codegen
mkdir build
pushd build
cmake ..
make 
ln -s ../Constant
./main_test > ${PWD}/end2end_VM_${align}_N_${remain}_M_32.txt
popd
popd
popd
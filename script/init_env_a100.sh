HOME_DIR=`pwd`
source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact

cd ~/SparTA && pip install -e .
cd $HOME_DIR
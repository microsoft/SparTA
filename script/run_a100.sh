pushd script/figure8 && bash run.sh 
popd

pushd script/figure13 && bash run.sh 
popd

mkdir -p results
cp script/figure*/*.pdf results/
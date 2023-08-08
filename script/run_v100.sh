pushd script/figure9 && bash run.sh
popd

pushd script/figure10 && bash run.sh
popd

pushd script/figure11 && bash run.sh
popd

pushd script/figure12 && bash run.sh
popd

pushd script/figure14 && bash run.sh
popd

pushd script/figure15 && bash run.sh
popd

pushd script/figure16 && bash run.sh
popd

pushd script/figure17 && bash run.sh
popd

mkdir -p results
cp script/figure*/*.pdf results/
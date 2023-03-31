+pushd ../baseline
+bash run.sh
+popd
python baseline_result.py --prefix ../baseline/log > baseline.txt

#!/bin/bash

if pip list |grep SparTA |grep 1.0
then
    echo "block_size,sparsity,latency" > sparta_results.csv
    for sparsity in 0.5 0.9 0.95 0.99
    do
        python sparta/test_sparta.py $sparsity 4096 4096 4096 32 1
    done
    for sparsity in 0.5 0.9 0.95 0.99
    do
        python sparta/test_sparta.py $sparsity 4096 4096 4096 1 64
    done
    for sparsity in 0.5 0.9 0.95 0.99
    do
        python sparta/test_sparta.py $sparsity 4096 4096 4096 32 64
    done
else
    echo "Failed: SparTA is not installed."
fi

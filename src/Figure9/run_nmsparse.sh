sparsity_ratio=(0.5 0.75 0.9)
# M1 M2 M3 M4 M5 M6 M7 M8
# m 1 1 1 1 1 1 1 1
# k 1024 2048 4096 8192 1024 4096 5120 20480
# n 1024 2048 4096 8192 4096 1024 20480 5120
shape_config=('M1 1 1024 1024'
              'M2 1 2048 2048'
              'M3 1 4096 4096'
              'M4 1 8192 8192'
              'M5 1 1024 4096'
              'M6 1 4096 1024'
              'M7 1 5120 20480'
              'M8 1 20480 5120'
            )


for sparsity in ${sparsity_ratio[@]}
do
    for info in "${shape_config[@]}";
    do
        name=`echo $info | awk '{print $1}'`
        m=`echo $info | awk '{print $2}'`
        k=`echo $info | awk '{print $3}'`
        n=`echo $info | awk '{print $4}'`
        echo $name $m $k $n $sparsity
        python ../nmsparse/run_SPMV_EW.py --sparsity_ratio $sparsity --name $name --M $m --K $k --N $n >> ./nmsparse_result.txt
    done
done

import re
from operator import itemgetter

with open("/workspace/v-leiwang3/SparTA_raw/src/Figure10/baseline_result.txt", "r") as f:
    file_contents = f.readlines()

benchmarks = []
for line in file_contents:
    match = re.search(r"SpMM sparsity ratio=(\d+\.\d+). shape=(M\d+) kernel=(\w+) latency=(\d+\.\d+)", line)
    if match:
        ratio, shape, kernel, latency = match.groups()
        shape_number = int(shape[1:])  
        benchmarks.append({"ratio": float(ratio), "shape": shape, "shape_number": shape_number, "kernel": kernel, "latency": float(latency)})

sorted_benchmarks = sorted(benchmarks, key=itemgetter("ratio", "shape_number"))


for benchmark in sorted_benchmarks:
    # if ratio = 0.5 
    # ratio = 0.5
    # ratio = 0.75
    ratio = 0.91
    # kernel = "sputnik"
    kernel = "cusparseblockELL"
    if benchmark["ratio"] == ratio and benchmark["kernel"] == kernel:
        print(benchmark["latency"]) 
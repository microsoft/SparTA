import re
from operator import itemgetter

with open("/workspace/v-leiwang3/SparTA/src/Figure9/baseline_result.txt", "r") as f:
    file_contents = f.readlines()

benchmarks = []
for line in file_contents:
    match = re.search(r"SpMV sparsity ratio=(\d+\.\d+). shape=(M\d+) kernel=(\w+) latency=(\d+\.\d+)", line)
    if match:
        ratio, shape, kernel, latency = match.groups()
        benchmarks.append({"ratio": float(ratio), "shape": shape, "kernel": kernel, "latency": float(latency)})

sorted_benchmarks = sorted(benchmarks, key=itemgetter("ratio", "shape"))

for benchmark in sorted_benchmarks:
    # if ratio = 0.5 
    ratio = 0.5
    kernel = "sputnik"
    if benchmark["ratio"] == ratio and benchmark["kernel"] == kernel:
        print(benchmark["latency"]) 
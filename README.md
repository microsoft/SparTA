## Overview

This branch is for the Mlsys'22 artifact evaluation of paper "EFFICIENT GPU KERNELS FOR N:M-SPARSE WEIGHTS IN DEEP LEARNING". 


## Evaluation Setup

* Artifacts Available:
The source code of Sparta is available at: https://github.com/microsoft/SparTA

* Artifacts Functional:
Documentation: the following document includes detailed guidelines on how to build, install, test NMSparse and the experiments to compare with other baselines.



## Environment setup

First, git clone the source code.
```
git clone https://github.com/microsoft/SparTA
cd SparTA && git checkout nmsparse_artifact
```
To make the reproducing easier, we provide a docker image that contains all dependencies and baselines. Build the docker image:
```
cd image
sudo docker build . -t artifact
```
Third, start a docker instance
```
sudo docker run -it --gpus all --shm-size 16G artifact
```
Following commands are executed in the docker.
First, we also need get the source code and initialize the environment.
```
# get source codes and scripts in the docker container
mkdir workspace && cd workspace
git clone -b nmsparse_artifact https://github.com/microsoft/SparTA.git
conda activate artifact
```
Then, we can run the artifacts in each folder.
```
# navigate to src directory
cd ./SparTA/src
# run SpMV experiment in Figure9
cd Figure9
bash run_baseline.sh
bash run_nmsparse.sh
# run SpMM on CudaCore experiment in Figure10
cd Figure10
bash run_baseline.sh
bash run_nmsparse.sh
# run SpMM on TensorCore experiment in Figure11
cd Figure11
bash run_baseline.sh
bash run_nmsparse.sh
# run end2end experiment in Figure12
cd Figure12
bash run.sh
```

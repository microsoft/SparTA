# FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu18.04
# FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
# Install tools and dependencies.
ENV PATH=${PATH}:/usr/local/cuda/bin/
RUN apt-get -y update --fix-missing
# for tzdata
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get install -y \
  emacs \
  git \
  wget \
  libgoogle-glog-dev

RUN apt-get update -y && \
    apt-get install -y libsndfile1

# Setup to install the latest version of cmake.
RUN apt-get install -y software-properties-common && \
    apt-get update && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update && apt-get install -y cmake unzip
# Set the working directory.
WORKDIR /root

#install sputnik
RUN git clone --recurse-submodules https://github.com/zheng-ningxin/sputnik.git && \
    cd sputnik && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DCUDA_ARCHS="70;75" && \
    make -j && cp sputnik/libspmm.so /usr/local/lib/ && cp -r /root/sputnik/third_party/abseil-cpp/absl /usr/local/include/

# install anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh && \
    bash Anaconda3-2021.11-Linux-x86_64.sh -b -p /root/anaconda && \
    eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda create -n artifact python=3.8 -y && \
    conda activate artifact

# install tvm
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.0/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz && \
    tar -xvf clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
RUN eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda activate artifact && \
    git clone --recursive https://github.com/linbinskn/tvm.git tvm && cd tvm && git checkout cuda_old && \
    apt-get update && apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev && \
    cd build && cmake .. && make -j &&  \
    pip install decorator

# install python packages
RUN eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda activate artifact && \
    pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116 && \
    pip install tensorboard onnxruntime graphviz onnx soundfile datasets==2.0.0 ply matplotlib numpy librosa xgboost==1.5.0 && \
    pip install nvitop accelerate cchardet seaborn triton==2.0.0.dev20221030 evaluate


RUN eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda activate artifact && echo "import torch\nprint(torch.cuda.is_available())" > tmp.py && python tmp.py > torch_cuda.log


# install PIT
RUN git clone https://github.com/zheng-ningxin/SparTA.git && cd SparTA && git checkout pit_sm70 && \
    eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda activate artifact && python setup.py develop

# install azcopy
RUN wget https://aka.ms/downloadazcopy-v10-linux && tar xzvf downloadazcopy-v10-linux && cp azcopy_linux_amd64_10.19.0/azcopy /usr/local/bin

# install transformers
RUN eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda activate artifact && \
    pip install git+https://github.com/iofu728/transformers.git@v4.26.0_PIT && mkdir /data/

# configure the bashrc
RUN echo 'export PYTHONPATH=/root/tvm/python:$PYTHONPATH \n\
export CUDA_HOME=/usr/local/cuda \n\
export PATH=/usr/local/cuda/bin:$PATH \n\
export TRANSFORMERS_CACHE=/data/.cache \n\
source ~/anaconda/etc/profile.d/conda.sh \n\
' >> /root/.bashrc

# install open-mpi
RUN eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda activate artifact && wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz && \
    tar -zxvf openmpi-4.1.5.tar.gz && cd openmpi-4.1.5 && ./configure --enable-shared --prefix=/usr/local/mpich && make -j && make install && \
    env MPICC=/usr/local/mpich/bin/mpicc python -m pip install mpi4py

RUN apt-get install -y ninja-build

# install mkl
RUN eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda activate artifact && pip install ninja && \
    cd /tmp && wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' && \
    apt-get update && apt --fix-broken install && apt-get install -y intel-mkl-64bit-2018.2-046

# Turbo
RUN eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda activate artifact &&  \
    git clone https://github.com/Tencent/TurboTransformers --recursive && \
    cd TurboTransformers && sed '/ctest/d' tools/build_and_run_unittests.sh > tools/build.sh && \
    sh tools/build.sh $PWD -DWITH_GPU=ON && pip install OpenNMT-py==2.3.0

# DeepSpeed
RUN eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda activate artifact && pip install pydantic==1.0 && pip install git+https://github.com/iofu728/deepspeed.git@pit_artifact

# Longformer
RUN eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda create --name longformer python=3.7 -y && conda activate longformer && \
    pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    conda install cudatoolkit=10.0 -y && pip install protobuf==3.20.0 datasets nvitop triton==2.0.0.dev20221030 deepspeed seaborn && \
    cd /tmp && git clone https://github.com/zheng-ningxin/SparTA.git && cd SparTA && git checkout pit_sm70 && python setup.py develop && \
    pip install git+https://github.com/iofu728/longformer.git && pip install git+https://github.com/iofu728/fairseq.git@v0.10.2_PIT && \
    pip install setuptools==59.5.0 protobuf==3.20.0

# nn_pruning
RUN eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda activate artifact && pip install git+https://github.com/iofu728/nn_pruning.git
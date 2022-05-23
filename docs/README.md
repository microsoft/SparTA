# SparTA Architecture Design

`SparTA` is an end-to-end system to harvest the speeding up gain from the model sparsity. Given a deep neural network (DNN) model and its initial *TeSA* (Tensor with Sparsity Attributes) specification, SparTA could generate the execution codes that are accelerated by the most appropriate ways to leverage its sparsity.

![arch](medias/arch.png)

## Key components

### SparTA propagator
SparTA leverages the [NNI](github.com/microsoft/nni) IR utilities to propagate the *TeSA* across the computational graph. 

After *TeSA* propagation, we could get a compact model through a simple structured pruner. It could be passed to the bottom layers or accessed via the *to-be-determined* API for users.

### Compiler interface
There are lots of compilation works for deep learning. SparTA focuses on sparsity acceleration, and don't want to duplicate the essential compiling functions. It leverages existing deep learning compilers (with essential modification) to implement

1. partition the whole model into sub-graphs after high level graph optimization (e.g., operator fusion).
2. combine individual kernel implementations generated from lower layer to a runnable model 
3. make necessary decisions to select the most suitable kernel implementations from many

#### PyTorch
For better support sparse training, `PyTorch` is a necessary compiler, though it has little graph level optimization and mainly works as a operator parser in this scenario. After specializing the sparse kernels, PyTorch could load them as custom operators and generate efficient sparse modules (e.g., [sparse attention]() module).

#### Rammer
Rammer ([nnFusion](github.com/microsoft/nn-fusion)) is one of the SOTA DNN compilers that is also open sourced by Microsoft. Ideally, Rammer gives the fused sub-graph IR that contains the input/output tesa and format requirements to SparTA via the compiler interface. SparTA perform the transformation and specialization for the sub-graph and return the transformed graph IR and specilizad kernel.

### SparTA specializer
SparTA specializer apply policies on the received subgraph (or fused kernel) and transform it into one or more highly efficient computing kernels. Such computing kernels could be both
- manually optimized computing templates and libraries
- automatically tuned kernels via tools like TVM 

Though there are still some questions still worth answering, such as

1. the boundary between compiler interface and specializer. For example, the granularity of specializer's input is a subgraph (specializer do fusion itself) or a fused kernel specification?
2. how to balance the benefits from the manually optimized kernel implementations and scalable automatic kernel generation.

A most feasible solution is leveraging similar to [JSON operator config in nnFusion](https://github.com/microsoft/nnfusion/blob/master/src/contrib/custom_op/README.md) but with more sparsity-related attributes, such as 
- the format (dense or sparse, including storage layout) of inputs and outputs
- links to template or library or TVM tuning command.

## Key abstractions

### *TeSA* 
TBA.

### Operators
TBA.

## Road map

1. PyTorch as compiler and manually optimized templates as bottom level computing library
2. integrate Rammer
3. bridge with TVM
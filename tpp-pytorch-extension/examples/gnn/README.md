Graph Neural Network
====================

This directory contains examples for two main GNN training workloads - GraphSAGE and Graph Attention Networks (GAT). Both these workloads use optimizations implemented in this extension for the MLP portion of the workloads. We have implemented optimizations for both FP32 and BF16. The two workload implementations are compatible with DGL version 0.8 and 0.9.

Pre-requisites: 
===============

1. gcc version 10.x or later

2. [DGL version 0.8+](https://github.com/dmlc/dgl)

```
Choose one of 0.8.x or 0.9.x branches
git clone https://github.com/dmlc/dgl.git
cd dgl
git submodule update --init --recursive

mkdir build
cd build
cmake -DUSE_LIBXSMM=ON ..
make -j

cd ../python
python setup.py install
```

3. To be able to automatically download datasets from OGB (https://ogb.stanford.edu)
```
pip install ogb
```
When you run the application in either [GraphSAGE](graphsage) or [GAT](gat) directories, the OGB module will automatically download the named dataset from the above link.

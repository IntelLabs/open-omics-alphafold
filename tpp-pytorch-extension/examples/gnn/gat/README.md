Graph Attention Netowrk
=======================

This example trains GAT model on OGBN-Products and OGBN-Papers100M on CPUs. It uses the optimizations in DGL as well as those in this extension for the MLP part of GNN training. 

Setup environment
=================

Install conda env and activate it as described in this [README](../../../README.md).

Install common GNN dependencies as described in this [README](../README.md).

To recompile the extension:

```
$make -C ../../.. reinstall
```

Training the model with OGBN-Products
=====================================

For FP32 training

To run baseline
```
$bash ./run.sh ogbn-products
```
To run optimized version
```
$bash ./run.sh ogbn-products --opt_mlp
```
For BF16 training (works only with optimized version)
```
$bash ./run.sh ogbn-products --opt_mlp --use_bf16
```
FP32 accuracy with optimized version on Intel® Xeon® Platinum 8380 server: 78.x % (SOTA)

Training the model with OGBN-Papers100M
=======================================

For FP32 training

To run baseline
```
$bash ./run.sh ogbn-papers100M
```
To run optimized version
```
$bash ./run.sh ogbn-papers100M --opt_mlp
```
For BF16 training (works only with optimized version)
```
$bash ./run.sh ogbn-papers100M --opt_mlp --use_bf16
```

FP32 accuracy with optimized version on Intel® Xeon® Platinum 8380 server: 65.x % (SOTA)

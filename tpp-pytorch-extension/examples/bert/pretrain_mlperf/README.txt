
Prerequsite:
-----------

Install and activate conda environment as described in README file here (TODO add link).

Install task specific requirements (one time):
$pip install -r requirements.txt

Create a link to path to MLPerf BERT pretraining dataset:
$ln -s <path to dataset> ./mlperf_dataset

Create a link to path to MLPerf BERT pretraining checkpoint:
$ln -s <path to checkpoint> ./ref_checkpoint

To run reference code (HuggingFace Transformers+PyTorch FP32 code) bert pretraining on single socket simply run:
$bash cmd.sh

For Optimized FP32 code:
$bash cmd.sh --use_tpp

For Optimized BF16 code:
$bash cmd.sh --use_tpp --tpp_bf16

You can enable unpad optimization by adding --unpad to optimized command line (FP32 or BF16).

To enable cpp profiling in extension add "--profile" to above command line.

To run bert pretraining on multi-node run:
$run_dist.sh -n <num_ranks> -ppn <ranks_per_node> -f hostfile bash cmd.sh --use_tpp --tpp_bf16 --unpad

By Default, script uses oneCCL backend for communication, use --dist_backend=mpi to use MPI backend.

To use optimized LAMB optimizer add --dist_lamb



Prerequsite:
-----------

Install and activate conda environment as described in README file here (TODO add link).

Install squad task specific requirements (one time):
$pip install -r requirements.txt

Create a link to path to SQUAD1 dataset:
$ln -s <path to dataset> ./SQUAD1

To run reference code (HuggingFace Transformers+PyTorch FP32 code) bert squad on single socket simply run:
$bash cmd.sh

For Optimized FP32 code:
$bash cmd.sh --use_tpp

For Optimized BF16 code:
$bash cmd.sh --use_tpp --tpp_bf16

You can enable unpad optimization by adding --unpad to optimized command line (FP32 or BF16).

For bit accurate BF8 (152) emulation code (only supported with unpad):
$bash cmd.sh --use_tpp --tpp_bf8 --unpad

To enable autograd profiling add "--profile" to above command line.

To run bert squad on multi-node run:
$run_dist.sh -n <num_ranks> -ppn <ranks_per_node> -f hostfile bash cmd.sh --use_tpp --tpp_bf16


Download SQUAD dataset:
----------------------

$dataset=SQUAD1
$( mkdir -p $dataset && cd $dataset && wget https://data.deepai.org/squad1.1.zip && unzip squad1.1.zip )


Download Squad fine-tuned model for inference:
---------------------------------------------
$bash download_squad_fine_tuned_model.sh

To run squad inference task:
$bash cmd_infer.sh --use_tpp --tpp_bf16 

To run squad inference in BF8:
$bash cmd_infer.sh --use_tpp --tpp_bf8 --unpad


# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

### input env params
root_condaenv=$1     # e.g. /home/<your-username>/anaconda3/envs/iaf2, root path of anaconda environment
root_home=$2         # e.g. /home/your-username, root path that holds all intermediate IO data
input_dir=$3         # e.g. $root_home/samples, path of all query .fa files (sequences in fasta format)
out_dir=$4           # e.g. $root_home/experiments/<experiment_name>, path that contains intermediates output of preprocessing, model inference, and final result
model_names=$5       # e.g. model_1_multimer_v3, the chosen model name of Alphafold2-multimer, or a comma seperated list of models "model_1_multimer_v3,model_2_multimer_v3,model_3_multimer_v3" (no spaces)
AF2_BF16=$6          # e.g. 1, Set to 1 to run code in BF16, 0 to run in FP32
num_multimer_predictions_per_model=$7     # e.g. 1, number of multimer predictions per model
random_seed=$8       # e.g. 123, random seed for model inference

if [ -z "$AF2_BF16" ]; then
  AF2_BF16=1
fi
if [ -z "$num_multimer_predictions_per_model" ]; then
  num_multimer_predictions_per_model=1
fi
if [ -z "$random_seed" ]; then
  random_seed=123
fi

data_dir=$root_data
log_dir=$root_home/logs
suffix=".fa"
n_sample=`ls ${input_dir}|grep ${suffix}|wc -l`
n_core=`lscpu|grep "^Core(s) per socket"|awk '{split($0,a," "); print a[4]}'`
n_socket=`lscpu|grep "^Socket(s)"|awk '{split($0,a," "); print a[2]}'`
((core_per_instance=$n_core*$n_socket))
((n_sample_0=$n_sample-1))
((core_per_instance_0=${core_per_instance}-1))
script="python run_modelinfer_pytorch_jit_multimer.py"
root_params=$root_home/weights/extracted/
workdir=`pwd`
if [ ! -d ${out_dir} ]; then
  echo "# <ERROR> No preprocessing result yet. You need to run xxx_preproc_baremetal.sh first. exiting"
  exit
fi

export TF_CPP_MIN_LOG_LEVEL=3
export LD_LIBRARY_PATH=$root_condaenv/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$root_condaenv/lib/libjemalloc.so:$LD_PRELOAD
# export KMP_AFFINITY=granularity=fine,compact,1,0 # 
# export KMP_BLOCKTIME=0
# export KMP_SETTINGS=0
export OMP_NUM_THREADS=$core_per_instance
export TF_ENABLE_ONEDNN_OPTS=1
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export USE_OPENMP=1
export USE_AVX512=1
export IPEX_ONEDNN_LAYOUT=1
export PYTORCH_TENSOREXPR=0
export CUDA_VISIBLE_DEVICES=-1

export AF2_BF16=$AF2_BF16                             # Set to 1 to run code in BF16, 0 to run in FP32

for f in `ls ${input_dir}|grep ${suffix}`; do
  fpath=${input_dir}/${f}
  # echo modelinfer ${fpath} on core 0-${core_per_instance_0} of socket 0-1
  # numactl -C 0-${core_per_instance_0} -m 0,1 $script \
  $script \
    --fasta_paths=${fpath} \
    --output_dir=${out_dir} \
    --model_names=${model_names} \
    --root_params=${root_params} \
    --random_seed=${random_seed} \
    --num_multimer_predictions_per_model=${num_multimer_predictions_per_model}

done
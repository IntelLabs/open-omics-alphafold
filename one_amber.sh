### input params
root_home=$1     # e.g. /home/your-username, root path that holds all intermediate IO data
input_dir=$2     # e.g. $root_home/samples, path of all query .fa files (sequences in fasta format)
out_dir=$3       # e.g. $root_home/experiments/<experiment_name>, path that contains intermediates output of preprocessing, model inference, and final result
model_names=$4   # e.g. model_1, the chosen model name of Alphafold2, or a comma seperated list of models "model_1,model_2,model_3,model_4,model_5" (no spaces)
model_preset=$5  # e.g. monomer or multimer, the chosen model preset of Alphafold2
num_multimer_predictions_per_model=$6   # e.g. 1, number of multimer predictions per model
random_seed=$7   # e.g. 0, random seed for reproducibility

if [ -z "$num_multimer_predictions_per_model" ]; then
  num_multimer_predictions_per_model=1
fi
if [ -z "$random_seed" ]; then
  random_seed=123
fi

log_dir=${root_home}/logs
suffix=".fa"
n_sample=`ls ${input_dir}|grep ${suffix}|wc -l`
n_core=`lscpu|grep "^Core(s) per socket"|awk '{split($0,a," "); print a[4]}'`
n_socket=`lscpu|grep "^Socket(s)"|awk '{split($0,a," "); print a[2]}'`
((core_per_instance=$n_core*$n_socket))
((n_sample_0=$n_sample-1))
((core_per_instance_0=${core_per_instance}-1))
script='run_amber.py'
# root_params=${root_home}/weights/extracted/${model_name}

for f in `ls ${input_dir}|grep ${suffix}`; do
	echo modelinfer ${input_dir}/${f}
	python $script \
		--fasta_paths ${input_dir}/${f} \
		--output_dir ${out_dir} \
		--model_names=${model_names} \
		--model_preset=${model_preset} \
		--num_multimer_predictions_per_model=${num_multimer_predictions_per_model} \
		--random_seed=${random_seed}

done
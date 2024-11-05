### input params
root_home=$1
data_dir=$2
input_dir=$3
out_dir=$4
model_name=$5

log_dir=${root_home}/logs
suffix=".fa"
n_sample=`ls ${input_dir}|grep ${suffix}|wc -l`
n_core=`lscpu|grep "^Core(s) per socket"|awk '{split($0,a," "); print a[4]}'`
n_socket=`lscpu|grep "^Socket(s)"|awk '{split($0,a," "); print a[2]}'`
((core_per_instance=$n_core*$n_socket))
((n_sample_0=$n_sample-1))
((core_per_instance_0=${core_per_instance}-1))
script='run_amber.py'
root_params=${root_home}/weights/extracted/${model_name}

for f in `ls ${input_dir}|grep ${suffix}`; do
	echo modelinfer ${input_dir}/${f}
	python $script \
		--fasta_paths ${input_dir}/${f} \
		--output_dir ${out_dir} \
		--model_names=${model_name} \
		--root_params=${root_params}
done
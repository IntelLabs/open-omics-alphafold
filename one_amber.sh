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
	--n_cpu $core_per_instance \
		--fasta_paths ${input_dir}/${f} \
		--output_dir ${out_dir} \
		--bfd_database_path=${data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
		--model_names=${model_name} \
		--root_params=${root_params} \
		--uniclust30_database_path=${data_dir}/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
		--uniref90_database_path=${data_dir}/uniref90/uniref90.fasta \
		--mgnify_database_path=${data_dir}/mgnify/mgy_clusters_2022_05.fa \
		--pdb70_database_path=${data_dir}/pdb70/pdb70 \
		--template_mmcif_dir=${data_dir}/pdb_mmcif/mmcif_files \
		--data_dir=${data_dir} \
		--max_template_date=2022-01-01 \
		--obsolete_pdbs_path=${data_dir}/pdb_mmcif/obsolete.dat \
		--hhblits_binary_path="$PWD/hh-suite/build/release/bin/hhblits" \
		--hhsearch_binary_path="$PWD/hh-suite/build/release/bin/hhsearch" \
		--jackhmmer_binary_path="$PWD/hmmer/release/bin/jackhmmer" \
		--kalign_binary_path=`which kalign`
done
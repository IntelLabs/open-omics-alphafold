import subprocess
import os
import time
import multiprocessing_functions as mpf

from absl import app
from absl import flags
from absl import logging

flags.DEFINE_string('root_home', None, 'home directory')
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('input_dir', None, 'root directory holding all .fa files')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will store the results.')

FLAGS = flags.FLAGS

script = "python run_preprocess_multimer.py"
base_fold_cmd = "/usr/bin/time -v {} \
                --n_cpu={} \
                --fasta_paths={} \
                --output_dir={} \
                --bfd_database_path={}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
                --uniref30_database_path={}/uniref30/UniRef30_2021_03 \
                --model_preset=multimer \
                --pdb_seqres_database_path={}/pdb_seqres/pdb_seqres.txt \
                --uniprot_database_path={}/uniprot/uniprot.fasta \
                --uniref90_database_path={}/uniref90/uniref90.fasta \
                --mgnify_database_path={}/mgnify/mgy_clusters_2022_05.fa \
                --template_mmcif_dir={}/pdb_mmcif/mmcif_files \
                --data_dir={} \
                --max_template_date=2022-01-01 \
                --obsolete_pdbs_path={}/pdb_mmcif/obsolete.dat \
                --hhblits_binary_path=$PWD/hh-suite/build/release/bin/hhblits \
                --hhsearch_binary_path=$PWD/hh-suite/build/release/bin/hhsearch \
                --hmmsearch_binary_path=$PWD/hmmer/release/bin/hmmsearch \
                --hmmbuild_binary_path=$PWD/hmmer/release/bin/hmmbuild \
                --jackhmmer_binary_path=$PWD/hmmer/release/bin/jackhmmer \
                --kalign_binary_path=`which kalign` \
                --run_in_parallel=true \
                "

def bash_subprocess(file_path, mem, core_list):
  """Starts a new bash subprocess and puts it on the specified cores."""

  data_dir = FLAGS.data_dir
  out_dir = FLAGS.output_dir
  log_dir = FLAGS.root_home + "/logs/"

  n_cpu = str(len(core_list))
  command = base_fold_cmd.format(script, n_cpu, file_path, out_dir, data_dir, data_dir, data_dir, data_dir, data_dir, data_dir, data_dir, data_dir, data_dir)
  numactl_args = ["numactl", "-m", mem, "-C", "-".join([str(core_list[0]), str(core_list[-1])]), command]

  print(" ".join(numactl_args))
  with open(log_dir + 'pre_log_' + os.path.basename(file_path) + '.txt', 'w') as f:
    try:
      process = subprocess.call(" ".join(numactl_args), shell=True, universal_newlines=True, stdout=f, stderr=f)
    except Exception as e:
      print('exception for', os.path.basename(file_path), e)
  return (process, file_path, mem, core_list)


def main(argv):
  t1 = time.time()
  
  input_dir = FLAGS.input_dir

  """The main function."""
  directory = input_dir
  total_cores = mpf.get_total_cores()
  print("Total cores: ", total_cores)
  print("Total memory: {} MB ".format(mpf.check_available_memory()))
  # Get the list of files in the directory.
  files = os.listdir(directory)
  for i, file in enumerate(files):
    files[i] = os.path.join(directory, file)

  MIN_MEM_PER_PROCESS=50*1024  # 64 GB
  MIN_CORES_PER_PROCESS=4
  LOAD_BALANCE_FACTOR=1

  max_processes_list = mpf.create_process_list(files, MIN_MEM_PER_PROCESS, MIN_CORES_PER_PROCESS, LOAD_BALANCE_FACTOR)
  files = mpf.start_process_list(files, max_processes_list, bash_subprocess)
  
  print("Following protein files couldn't be processed")
  print(files)
  t2 = time.time()
  print('### total preprocessing time: %d sec' % (t2-t1))

if __name__ == "__main__":
  flags.mark_flags_as_required([
      'root_home',
      'data_dir',
      'input_dir',
      'output_dir',
  ])
  app.run(main)
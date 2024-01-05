import subprocess
import os
import psutil
import time
import multiprocessing as mp


from absl import app
from absl import flags
from absl import logging

flags.DEFINE_string('root_condaenv', None, 'conda environment directory path')
flags.DEFINE_string('root_home', None, 'home directory')
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('input_file', None, 'file containing list of full path of all .fa files - one line per file.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will store the results.')
flags.DEFINE_string('model_name', None, 'Names of models to use')
flags.DEFINE_integer('AF2_BF16', 1, 'Set to 0 for FP32 precision run.')
flags.DEFINE_string('n_cpu', None, 'number of threads to use')
flags.DEFINE_string('error_file', None, 'file containing list of full path of all the failed .fa files.')
FLAGS = flags.FLAGS

script = "python run_modelinfer_pytorch_jit.py"
base_fold_cmd = "/usr/bin/time -v {} \
                --n_cpu {} \
                --fasta_paths {} \
                --output_dir {} \
                --bfd_database_path={}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
                --model_names={} \
                --root_params={} \
                --uniclust30_database_path={}/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
                --uniref90_database_path={}/uniref90/uniref90.fasta \
                --mgnify_database_path={}/mgnify/mgy_clusters.fa \
                --pdb70_database_path={}/pdb70/pdb70 \
                --template_mmcif_dir={}/pdb_mmcif/mmcif_files \
                --data_dir={} \
                --max_template_date=2022-01-01 \
                --obsolete_pdbs_path={}/pdb_mmcif/obsolete.dat \
                --hhblits_binary_path=$PWD/hh-suite/build/release/bin/hhblits \
                --hhsearch_binary_path=$PWD/hh-suite/build/release/bin/hhsearch \
                --jackhmmer_binary_path=$PWD/hmmer/release/bin/jackhmmer \
                --kalign_binary_path=`which kalign` \
                "

def start_bash_subprocess(file_path, n_cpu):
  """Runs a new bash subprocess"""
  data_dir = FLAGS.data_dir
  out_dir = FLAGS.output_dir
  root_params = FLAGS.root_home + "/weights/extracted/model_1"
  log_dir = FLAGS.root_home + "/logs/"
  model_name=FLAGS.model_name

  command = base_fold_cmd.format(script, n_cpu, file_path, out_dir, data_dir, model_name, root_params, data_dir, data_dir, data_dir, data_dir, data_dir, data_dir, data_dir)

  print (command)
  with open(log_dir + 'inference_log_' + os.path.basename(file_path) + '.txt', 'w') as f:
    try:
      process = subprocess.run(command, shell=True, universal_newlines=True, stdout=f, stderr=f)
    except Exception as e:
      print('exception for', os.path.basename(file_path), e)
  return process

def multiprotein_run(files, n_cpu):
  # Process the files one by one

  error_files = []
  for file in files:
    file_path = file
    process = run_bash_subprocess(file_path, n_cpu)
    if (process.returncode != 0):
        error_files.append(file_path)

  return error_files

def main(argv):
  t1 = time.time()

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  root_condaenv=FLAGS.root_condaenv
  input_file = FLAGS.input_file

  files = []
  # Get the list of files in the directory.
  with f = open(input_file, "r"):
    for line in f:
        files.append(line.strip())

  os.environ["LD_PRELOAD"] = "{}/lib/libiomp5.so:{}/lib/libjemalloc.so:{}".format(root_condaenv,root_condaenv,os.environ["LD_PRELOAD"])
  os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
  os.environ["MALLOC_CONF"] = "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
  os.environ["USE_OPENMP"] = "1"
  os.environ["USE_AVX512"] = "1"
  os.environ["IPEX_ONEDNN_LAYOUT"] = "1"
  os.environ["PYTORCH_TENSOREXPR"] = "0"
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  os.environ["AF2_BF16"] = str(FLAGS.AF2_BF16)

  """The main function."""

  os.environ["OMP_NUM_THREADS"] = str(n_cpu)
  print("Number of OMP Threads = {}".format(os.environ.get('OMP_NUM_THREADS')))
  returned_files = multiprotein_run(files, max_processes)
  print("Following protein files couldn't be processed with {} instances".format(max_processes))
  print(returned_files)
  with open(FLAGS.error_file, "w") as ef:
    for name in returned_files:
        ef.write(name + "\n")

  t2 = time.time()
  print('### Total inference time: %d sec' % (t2-t1))


if __name__ == "__main__":
  flags.mark_flags_as_required([
      'root_condaenv',
      'root_home',
      'data_dir',
      'input_file',
      'output_dir',
      'model_name',
      'n_cpu',
      'error_file'
  ])
  # main()
  app.run(main)

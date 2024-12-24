import subprocess
import os
import time
import multiprocess_functions as mpf

from absl import app
from absl import flags
from absl import logging

flags.DEFINE_string('root_condaenv', None, 'conda environment directory path')
flags.DEFINE_string('root_home', None, 'home directory')
flags.DEFINE_string('input_dir', None, 'root directory holding all .fa files')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will store the results.')
flags.DEFINE_string('model_names', None, 'Names of models to use')
flags.DEFINE_integer('AF2_BF16', 1, 'Set to 0 for FP32 precision run.')
FLAGS = flags.FLAGS

script = "python run_modelinfer_pytorch_jit.py"
base_fold_cmd = "/usr/bin/time -v {} \
                --fasta_paths {} \
                --output_dir {} \
                --model_names={} \
                --root_params={} \
                "

def bash_subprocess(file_path, mem, core_list):
  """Starts a new bash subprocess and puts it on the specified cores."""
  out_dir = FLAGS.output_dir
  root_params = FLAGS.root_home + "/weights/extracted/"
  log_dir = FLAGS.root_home + "/logs/"
  model_names=FLAGS.model_names

  command = base_fold_cmd.format(script, file_path, out_dir, model_names, root_params)
  numactl_args = ["numactl", "-m", mem, "-C", "-".join([str(core_list[0]), str(core_list[-1])]), command]

  print(" ".join(numactl_args))
  with open(log_dir + 'inference_log_' + os.path.basename(file_path) + '.txt', 'w') as f:
    try:
      process = subprocess.call(" ".join(numactl_args), shell=True, universal_newlines=True, stdout=f, stderr=f)
    except Exception as e:
      print('exception for', os.path.basename(file_path), e)
  return (process, file_path, mem, core_list)

def main(argv):
  t1 = time.time()

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  input_dir = FLAGS.input_dir

  os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
  os.environ["MALLOC_CONF"] = "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
  os.environ["USE_OPENMP"] = "1"
  os.environ["USE_AVX512"] = "1"
  os.environ["IPEX_ONEDNN_LAYOUT"] = "1"
  os.environ["PYTORCH_TENSOREXPR"] = "0"
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  os.environ["AF2_BF16"] = str(FLAGS.AF2_BF16)

  """The main function."""
  directory = input_dir
  total_cores = mpf.get_total_cores()
  print("Total cores: ", total_cores)
  print("Total memory: {} MB ".format(mpf.check_available_memory()))

  # Get the list of files in the directory.
  files = os.listdir(directory)
  for i, file in enumerate(files):
    files[i] = os.path.join(directory, file)
  
  MIN_MEM_PER_PROCESS=32*1024  # 32 GB
  MIN_CORES_PER_PROCESS=8
  LOAD_BALANCE_FACTOR=4

  max_processes_list = mpf.create_process_list(files, MIN_MEM_PER_PROCESS, MIN_CORES_PER_PROCESS, LOAD_BALANCE_FACTOR)
  files = mpf.start_process_list(files, max_processes_list, bash_subprocess)

  print("Following protein files couldn't be processed")
  print(files)
  t2 = time.time()
  print('### Total inference time: %d sec' % (t2-t1))


if __name__ == "__main__":
  flags.mark_flags_as_required([
      'root_home',
      'input_dir',
      'output_dir',
      'model_names'
  ])
  app.run(main)

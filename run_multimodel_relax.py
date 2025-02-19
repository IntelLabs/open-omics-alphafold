import subprocess
import os
import time
import multiprocess_functions as mpf 
from datetime import datetime
d = datetime.now()
timestamp = "relax_%04d%02d%02d%02d%02d" % (d.year, d.month, d.day, d.hour, d.minute)

from absl import app
from absl import flags
from absl import logging

flags.DEFINE_string('root_home', None, 'home directory')
flags.DEFINE_string('input_dir', None, 'root directory holding all .fa files')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will store the results.')
flags.DEFINE_string('model_names', None, 'Names of models to use')
flags.DEFINE_integer('num_multimer_predictions_per_model', 1, 'How many '
                     'predictions (each with a different random seed) will be '
                     'generated per model. E.g. if this is 2 and there are 5 '
                     'models then there will be 10 predictions per input. '
                     'Note: this FLAG only applies in multimer mode')
flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
FLAGS = flags.FLAGS

script = "python run_amber.py"
base_fold_cmd = "/usr/bin/time -v {} \
                --fasta_paths {} \
                --output_dir {} \
                --model_names={} \
                --model_preset={} \
                --random_seed={} \
                --num_multimer_predictions_per_model={} \
                "

def bash_subprocess(file_path, model_name, random_seed, mem, core_list):
  """Starts a new bash subprocess and puts it on the specified cores."""
  out_dir = FLAGS.output_dir
  root_params = FLAGS.root_home + "/weights/extracted/"
  log_dir = FLAGS.root_home + "/logs/" + str(timestamp) + "/"
  os.makedirs(log_dir, exist_ok=True) 
  # number_multimer_predictions_per_model = FLAGS.num_multimer_predictions_per_model
  number_multimer_predictions_per_model = 1
  model_preset = FLAGS.model_preset

  command = base_fold_cmd.format(script, file_path, out_dir, model_name, model_preset, random_seed, number_multimer_predictions_per_model)
  numactl_args = ["numactl", "-m", mem, "-C", "-".join([str(core_list[0]), str(core_list[-1])]), command]

  print(" ".join(numactl_args))
  with open(log_dir + 'relax_log_' + os.path.basename(file_path) + "_" + model_name + '_' + str(random_seed) + '.txt', 'w') as f:
    try:
      process = subprocess.call(" ".join(numactl_args), shell=True, universal_newlines=True, stdout=f, stderr=f)
    except Exception as e:
      print('exception for', os.path.basename(file_path), e)
  return (process, file_path, model_name, mem, core_list)


def main(argv):
  t1 = time.time()

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  input_dir = FLAGS.input_dir

  os.environ["USE_OPENMP"] = "1"

  """The main function."""
  directory = input_dir
  total_cores = mpf.get_total_cores()
  print("Total cores: ", total_cores)
  print("Total memory: {} MB ".format(mpf.check_available_memory()))

  # Get the list of files in the directory.
  files = os.listdir(directory)
  for i, file in enumerate(files):
    files[i] = os.path.join(directory, file)
  
  model_list = FLAGS.model_names.strip('[]').split(',')

  MIN_MEM_PER_PROCESS=16*1024  # 16 GB
  MIN_CORES_PER_PROCESS=1
  LOAD_BALANCE_FACTOR=1
  
  num_instances = len(files) * len(model_list) * FLAGS.num_multimer_predictions_per_model
  max_processes_list = mpf.create_process_list(num_instances, MIN_MEM_PER_PROCESS, MIN_CORES_PER_PROCESS, LOAD_BALANCE_FACTOR)
  print("max_processes_list", max_processes_list)
  error_combo = mpf.multiprocess_models(files, max_processes_list, model_list, FLAGS.num_multimer_predictions_per_model, bash_subprocess)

  print("Following protein combination couldn't be processed".format(error_combo))
  t2 = time.time()
  print('### Total Relaxation time: %d sec' % (t2-t1))


if __name__ == "__main__":
  flags.mark_flags_as_required([
      'root_home',
      'input_dir',
      'output_dir',
      'model_names',
      'model_preset',
  ])
  app.run(main)

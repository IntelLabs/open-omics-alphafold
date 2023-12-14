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
flags.DEFINE_string('input_dir', None, 'root directory holding all .fa files')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will store the results.')
flags.DEFINE_string('model_name', None, 'Names of models to use')
flags.DEFINE_integer('AF2_BF16', 1, 'Set to 0 for FP32 precision run.')
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

def start_bash_subprocess(file_path, mem, core_list):
  """Starts a new bash subprocess and puts it on the specified cores."""
  data_dir = FLAGS.data_dir
  out_dir = FLAGS.output_dir
  root_params = FLAGS.root_home + "/weights/extracted/model_1"
  log_dir = FLAGS.root_home + "/logs/"
  model_name=FLAGS.model_name

  n_cpu = str(len(core_list))
  command = base_fold_cmd.format(script, n_cpu, file_path, out_dir, data_dir, model_name, root_params, data_dir, data_dir, data_dir, data_dir, data_dir, data_dir, data_dir)
  numactl_args = ["numactl", "-m", mem, "-C", "-".join([str(core_list[0]), str(core_list[-1])]), command]

  print(" ".join(numactl_args))
  with open(log_dir + 'inference_log_' + os.path.basename(file_path) + '.txt', 'w') as f:
    try:
      process = subprocess.call(" ".join(numactl_args), shell=True, universal_newlines=True, stdout=f, stderr=f)
    except Exception as e:
      print('exception for', os.path.basename(file_path), e)
  return (process, file_path, mem, core_list)

def check_available_memory():
  """Checks for available memory using psutil."""
  mem = psutil.virtual_memory()
  available_memory = mem.available
  # print("Available memory: {} MB".format(available_memory / 1024 ** 2))
  return available_memory / 1024 ** 2

def get_file_size(file_path):
  """Gets the size of the file in bytes."""
  size = subprocess.check_output(["wc", "-c", file_path])
  size = int(size.decode("utf-8").split()[0])
  return size


def multiprocessing_run(files, max_processes):
  size_dict = dict()
  for file in files:
    size_dict[file] = get_file_size(file)

  # print(size_dict) 
  sorted_size_dict = dict(sorted(size_dict.items(), key=lambda item: item[1], reverse=True))
  # print(sorted_size_dict)
  total_cores = os.cpu_count()//2
  core_list = range(os.cpu_count()//2)
  cores_per_process = total_cores // max_processes
  pool = mp.Pool(processes=max_processes)

  queue = [i for i in range(max_processes)]
  error_files = []
  def update_queue(result):
    print(result)
    queue.append(result[3][0] // cores_per_process)
    if (result[0] != 0):
      error_files.append(result[1])

  # Iterate over the files and start a new subprocess for each file.
  print(len(sorted_size_dict))
  results = [None] * len(sorted_size_dict)

  #numa_nodes
  lscpu = subprocess.Popen(["lscpu"], stdout=subprocess.PIPE)
  grep = subprocess.Popen(["grep", "NUMA node(s):"], stdin=lscpu.stdout, stdout=subprocess.PIPE)
  awk = subprocess.Popen(["awk", "{print $3}"], stdin=grep.stdout, stdout=subprocess.PIPE)
  #Get the output
  numa_nodes = int(awk.communicate()[0])

  i = 0
  for file, value in sorted_size_dict.items():
    file_path = file
    # process_num = i % max_processes
    process_num = queue.pop(0)
    if process_num < max_processes//2:
      mem = '0'
    else:
      if numa_nodes > 1:
        mem = '1'
      else:
        mem = '0'
    
    if max_processes == 1:
      if numa_nodes > 1:
        mem = '0,1'
      else:
        mem = '0'
    
    results[i] = pool.apply_async(start_bash_subprocess, args=(file_path, mem, core_list[process_num*cores_per_process: (process_num+1)*cores_per_process]), callback = update_queue)
    i += 1
    while len(queue) == 0 and i < len(sorted_size_dict):
        time.sleep(0.05)
  pool.close()
  pool.join()

  return error_files

def main(argv):
  t1 = time.time()

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  root_condaenv=FLAGS.root_condaenv
  input_dir = FLAGS.input_dir

  os.environ["LD_PRELOAD"] = "{}/lib/libiomp5.so:{}/lib/libjemalloc.so:$LD_PRELOAD".format(root_condaenv, root_condaenv)
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
  total_cores = os.cpu_count()//2

  if total_cores % 16 == 0:
    max_processes_list = [16, 8, 4, 2, 1]
  elif total_cores % 8 == 0:
    max_processes_list = [8, 4, 2, 1]
  elif total_cores % 4 == 0:
    max_processes_list = [4, 2, 1]
  else:
    max_processes_list = [2, 1]
  print("Total cores: ", os.cpu_count() //2)
  print("Total memory: {} MB ".format(check_available_memory()))

  # Get the list of files in the directory.
  files = os.listdir(directory)
  for i, file in enumerate(files):
    files[i] = os.path.join(directory, file)

  for max_processes in max_processes_list:
    os.environ["OMP_NUM_THREADS"] = str(total_cores//max_processes)
    print("Number of OMP Threads = {}, for {} instances".format(os.environ.get('OMP_NUM_THREADS'), max_processes))
    if len(files) >= max_processes:
      returned_files = multiprocessing_run(files, max_processes)
      print("Following protein files couldn't be processed with {} instances".format(max_processes))
      print(returned_files)
    else:
      continue

    files = returned_files
  
  print("Following protein files couldn't be processed")
  print(files)
  t2 = time.time()
  print('### Total inference time: %d sec' % (t2-t1))


if __name__ == "__main__":
  flags.mark_flags_as_required([
      'root_condaenv',
      'root_home',
      'data_dir',
      'input_dir',
      'output_dir',
      'model_name'
  ])
  # main()
  app.run(main)
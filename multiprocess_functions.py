import subprocess
import os
import psutil
import multiprocessing as mp
import math
import time

def check_available_memory():
  """Checks for available memory using psutil."""
  mem = psutil.virtual_memory()
  available_memory = mem.available
  return available_memory / 1024 ** 2

def get_total_cores():
  #sockets
  lscpu = subprocess.Popen(["lscpu"], stdout=subprocess.PIPE)
  grep = subprocess.Popen(["grep", "Thread(s) per core:"], stdin=lscpu.stdout, stdout=subprocess.PIPE)
  awk = subprocess.Popen(["awk", "{print $4}"], stdin=grep.stdout, stdout=subprocess.PIPE)
  #Get the output
  thread_per_core = int(awk.communicate()[0])
  return os.cpu_count()//thread_per_core

def get_file_size(file_path):
  """Gets the size of the file in bytes."""
  size = subprocess.check_output(["wc", "-c", file_path])
  size = int(size.decode("utf-8").split()[0])
  return size

def get_core_list(cores_per_process):
  #numa_nodes
  lscpu = subprocess.Popen(["lscpu"], stdout=subprocess.PIPE)
  grep = subprocess.Popen(["grep", "NUMA node(s):"], stdin=lscpu.stdout, stdout=subprocess.PIPE)
  awk = subprocess.Popen(["awk", "{print $3}"], stdin=grep.stdout, stdout=subprocess.PIPE)
  #Get the output
  numa_nodes = int(awk.communicate()[0])

  core_min_max = []
  cores_in_numa = os.cpu_count()
  for i in range(numa_nodes):
    lscpu = subprocess.Popen(["lscpu"], stdout=subprocess.PIPE)
    grep = subprocess.Popen(["grep", "NUMA node" + str(i) + " CPU(s):"], stdin=lscpu.stdout, stdout=subprocess.PIPE)
    awk = subprocess.Popen(["awk", "{print $4}"], stdin=grep.stdout, stdout=subprocess.PIPE)  
    l = awk.communicate()[0].decode('utf-8').split(',')[0].split('-')
    l = [int(x) for x in l]
    cores_in_numa = min(cores_in_numa, l[1] - l[0] + 1)
    core_min_max.append(l)

  core_list = []
  if cores_per_process <= cores_in_numa:       # Normal case
    for i in range(numa_nodes):
      for j in range(cores_in_numa//cores_per_process):
        core_list = core_list + list(range(core_min_max[i][0] + j*cores_per_process, core_min_max[i][0] + (j+1)*cores_per_process))
  else:        # single process case or single socket case
    core_list = range(os.cpu_count()//2)

  return core_list, numa_nodes

def multiprocessing_run(files, max_processes, bash_subprocess):
  size_dict = dict()
  for file in files:
    size_dict[file] = get_file_size(file)

  sorted_size_dict = dict(sorted(size_dict.items(), key=lambda item: item[1], reverse=True))

  total_cores = get_total_cores()
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

  core_list, numa_nodes = get_core_list(cores_per_process)

  i = 0
  for file, value in sorted_size_dict.items():
    file_path = file
    process_num = queue.pop(0)

    if max_processes < numa_nodes:
      if max_processes == 1:
        if numa_nodes > 1:
          mem = '0-{}'.format(numa_nodes-1)
        else:
          mem = '0'
      else:
        mem = '{}-{}'.format(str(process_num * (numa_nodes//max_processes)), str(((process_num + 1) * (numa_nodes//max_processes)) - 1))
    else:
      mem = str(process_num//(max_processes//numa_nodes))
    
    results[i] = pool.apply_async(bash_subprocess, args=(file_path, mem, core_list[process_num*cores_per_process: (process_num+1)*cores_per_process]), callback = update_queue)
    i += 1
    while len(queue) == 0 and i < len(sorted_size_dict):
        time.sleep(0.05)
  pool.close()
  pool.join()

  return error_files

def create_process_list(files, MIN_MEM_PER_PROCESS, MIN_CORES_PER_PROCESS, LOAD_BALANCE_FACTOR):
  total_cores = get_total_cores()
  #numa_nodes
  lscpu = subprocess.Popen(["lscpu"], stdout=subprocess.PIPE)
  grep = subprocess.Popen(["grep", "NUMA node(s):"], stdin=lscpu.stdout, stdout=subprocess.PIPE)
  awk = subprocess.Popen(["awk", "{print $3}"], stdin=grep.stdout, stdout=subprocess.PIPE)
  #Get the output
  numa_nodes = int(awk.communicate()[0])
  cores_per_numa = total_cores//numa_nodes

  #sockets
  lscpu = subprocess.Popen(["lscpu"], stdout=subprocess.PIPE)
  grep = subprocess.Popen(["grep", "Socket(s):"], stdin=lscpu.stdout, stdout=subprocess.PIPE)
  awk = subprocess.Popen(["awk", "{print $2}"], stdin=grep.stdout, stdout=subprocess.PIPE)
  #Get the output
  sockets = int(awk.communicate()[0])

  memory_per_numa_domain = check_available_memory()/numa_nodes
  # Memory max
  mm = memory_per_numa_domain // MIN_MEM_PER_PROCESS
  # Core_max
  cm = cores_per_numa // MIN_CORES_PER_PROCESS

  pn = len(files) // numa_nodes
  # Load_balance_max
  lbm = max(pn//LOAD_BALANCE_FACTOR, 1)
  print("Memory max {}, Core max {}, Load Balance max {}".format(mm, cm, lbm))
  max_p = min(mm, cm, lbm)

  # number which is 2^x and less than or equal to max
  max_p = 2**int(math.log2(max_p))

  max_processes_list = []
  while max_p > 0:
    max_processes_list.append(max_p*numa_nodes)
    max_p = max_p // 2
  if numa_nodes != sockets:
    max_processes_list.append(sockets)
  if numa_nodes != 1:
    max_processes_list.append(1)
  print("Max processes list: ", max_processes_list)

  return max_processes_list

def start_process_list(files, max_processes_list, bash_subprocess):
  total_cores = get_total_cores()
  for max_processes in max_processes_list:
    os.environ["OMP_NUM_THREADS"] = str(total_cores//max_processes)
    print("Number of OMP Threads = {}, for {} instances".format(os.environ.get('OMP_NUM_THREADS'), max_processes))
    if len(files) >= max_processes:
      returned_files = multiprocessing_run(files, max_processes, bash_subprocess)
      print("Following protein files couldn't be processed with {} instances".format(max_processes))
      print(returned_files)
    else:
      continue

    files = returned_files
  return files
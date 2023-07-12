#!/bin/bash

### hyperparams zone start ###
root_home=$1        # path that holds all input/intermediates/output data
refdata_dir=$2      # path of alphafold reference dataset, including bfd, uniref, params etc.
conda_env_name=$3   # intel-alphafold2 conda env
experiment_name=$4  # project name (it may include multiple different samples)
model_name=model_1  # alphafold weight prefix selected for model inference
### hyperparams zone end   ###

# check if essential hyperparams are set
if [[ $root_home == "null" ]]; then
  echo "### <ERROR> plz provide a path that holds all input/intermediates/output data. exiting"
  exit
elif [[ $refdata_dir == "null" ]]; then
  echo "### <ERROR> plz provide a path of alphafold reference dataset, including bfd, uniref, params etc.. exiting"
  exit
fi

# check and create directories for I/O data
weights_dir=$root_home/weights
extracted_weights_dir=$weights_dir/extracted
log_dir=$root_home/logs
samples_dir=$root_home/samples
result_dir=$root_home/experiments
experiment_dir=$root_home/experiments/$experiment_name

if [ -d $root_home ]; then # validate home path
  if [ ! -d $samples_dir ]; then
    echo " ## <ERROR> path to input samples $samples_dir should not be empty. exiting"
    mkdir $samples_dir
    exit
  fi
  # enumerate and create every subfolder 
  for d in $weights_dir $extracted_weights_dir $log_dir $result_dir $experiment_dir; do
    if [ ! -d $d ]; then
      mkdir $d
    fi
  done
else # no access to home path
  echo "### <ERROR> invalid root_home directory: $root_home. exiting"
  exit
fi

# validate all reference data of alphafold2
ref_bfd_dir=$refdata_dir/bfd
ref_mgnify_dir=$refdata_dir/mgnify
ref_pdb70_dir=$refdata_dir/pdb70
ref_pdbmmcif_dir=$refdata_dir/pdb_mmcif
ref_uniclust30_dir=$refdata_dir/uniclust30
ref_uniref90_dir=$refdata_dir/uniref90
ref_params_dir=$refdata_dir/params
if [ -d $refdata_dir ]; then # validate reference dataset of alphafold2
  for d in $ref_bfd_dir $ref_mgnify_dir $ref_pdbmmcif_dir $ref_pdb70_dir $ref_uniclust30_dir $ref_uniref90_dir $ref_params_dir; do
    if [ ! -d $d ]; then
      echo " ## <ERROR> invalid reference data folder $d. exiting"
      exit
    fi
  done
  # validate bfd
  n_ffdata=`ls $ref_bfd_dir|grep "ffdata$"|wc -l`
  n_ffidx=`ls $ref_bfd_dir|grep "ffindex$"|wc -l`
  if [[ $n_ffdata != 3 ]]; then
    echo " ## <ERROR> incomplete ffdata in BFD dataset, plz check folder $ref_bfd_dir"
    exit
  fi
  if [[ $n_ffidx != 3 ]]; then
    echo " ## <ERROR> incomplete ffindex in BFD dataset, plz check folder $ref_bfd_dir. exiting"
    exit
  fi
  # validate mgnify
  n_fa=`ls $ref_mgnify_dir|grep "fa$"|wc -l`
  n_fasta=`ls $ref_mgnify_dir|grep "fasta$"|wc -l`
  if [ 0 -lt $n_fa ]; then
    f_mgnify=$ref_mgnify_dir/`ls $ref_mgnify_dir|grep "fa$"`
  elif [ 0 -lt $n_fasta ]; then
    f_mgnify=$ref_mgnify_dir/`ls $ref_mgnify_dir|grep "fasta$"`
  else
    echo " ## <ERROR> invalid mgnify dataset, plz check folder $ref_mgnify_dir. exiting"
    exit
  fi
  # validate pdb70
  n_pdb70_files=`ls $ref_pdb70_dir|wc -l`
  if [ $n_pdb70_files -lt 9 ]; then
    echo " ## <ERROR> incomplete pdb70 dataset, plz check folder $ref_pdb70_dir. exiting"
    exit
  fi
  # validate pdb_mmcif
  f_obs="$ref_pdbmmcif_dir/obsolete.dat"
  mmcif_dir="$ref_pdbmmcif_dir/mmcif_files"
  n_mmcif=`ls $mmcif_dir|wc -l`
  if [ $n_mmcif -lt 180000 ]; then
    echo " ## <ERROR> incomplete pdb mmcif dataset, plz check folder $mmcif_dir. exiting"
    exit
  elif [ ! -f $f_obs ]; then
    echo " ## <ERROR> missing $f_obs, plz check folder $ref_pdbmmcif_dir, exiting"
    exit
  fi
  # validate uniclust30
  uniclust30_dir=$ref_uniclust30_dir/`ls $ref_uniclust30_dir`
  n_uniclust30_files=`ls $uniclust30_dir|wc -l`
  if  [ $n_uniclust30_files -lt 13 ]; then
    echo " ## <ERROR> incomplete uniclust30 dataset, plz check folder $uniclust30_dir. exiting"
    exit
  fi
  # validate uniref90
  f_uniref90="$ref_uniref90_dir/uniref90.fasta"
  if [ ! -f $f_uniref90 ]; then
    echo " ## <ERROR> invalid uniref90 dataset, plz check folder $ref_uniref90_dir. exiting"
    exit
  fi
  # validate model params
  f_params="${ref_params_dir}/params_${model_name}.npz"
  if [ ! -f $f_params ]; then
    echo " ## <ERROR> invalid params folder $ref_params_dir or invalid model name $model_name. exiting"
    exit
  fi
else
  echo " ## <ERROR> invalid reference data path for alphafold2: $refdata_dir. exiting"
fi

# create conda env
echo " ## <INFO> installing alphafold2 conda env"
conda install -y -c conda-forge openmm pdbfixer
conda install -y -c bioconda hmmer hhsuite kalign2
conda install -y -c pytorch pytorch cpuonly
conda install -y jemalloc

# download source code
#if [ ! -d "intel-alphafold2" ]; then
#  git clone https://github.com/intelxialei/intel-alphafold2
#fi
#cd intel-alphafold2
export IAF2_HOME=`pwd`

# install pip dependencies
python -m pip install absl-py biopython chex dm-haiku dm-tree immutabledict jax ml-collections numpy scipy tensorflow pandas psutil tqdm joblib
python -m pip install --upgrade jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_releases.html

# install ipex[cpu]
python -m pip install intel-extension-for-pytorch

# compile Intel extension for TPP
cd $IAF2_HOME/tpp-pytorch-extension
git submodule update --init
python setup.py install
cd $IAF2_HOME

# extract model
echo "### <INFO> extracting model parameter file"
if [ ! -d $weights_dir ]; then
  mkdir $weights_dir
fi
if [ ! -d $extracted_weights_dir ]; then
  mkdir ${extracted_weights_dir}/${model_name}
fi
python extract_params.py --input $f_params --output_dir ${extracted_weights_dir}/${model_name}

echo "### <INFO> Initialization of intel-alphafold2 is done."

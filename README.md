# AlphaFold2 optimized on Intel Xeon CPU

This repository contains an inference pipeline of AlphaFold2 with a *bona fide* translation from *Haiku/JAX* (https://github.com/deepmind/alphafold) to PyTorch.

<u>**Declaration 1**</u>
Any publication that discloses findings arising from using this source code or the model parameters should [cite](#citing-this-work) the [AlphaFold paper](https://doi.org/10.1038/s41586-021-03819-2). Please also refer to the [Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf) for a detailed description of the method.

<u>**Declaration 2**</u>
The setup procedures were modified from the two repos:
   https://github.com/kalininalab/alphafold_non_docker
   https://github.com/deepmind/alphafold
with only some exceptions. I will label the difference for highlight.

<u>**Declaration 3**</u>
This repo is independently implemented, and is different from a previously unofficial version (https://github.com/lucidrains/alphafold2).
No one is better than the other, and the differences are in 3 points:
(1) this repo is major in acceleration of inference, in compatible to weights released from DeepMind;
(2) this repo delivers a reliable pipeline accelerated on Intel® Core/Xeon and Intel® Optane® PMem by Intel® oneAPI.
(3) this repo places CPU as its primary computation resource for acceleration, which may not provide an optimal speed on GPU.


## Primary solution for setup of intel-alphafold2 environment

1. install anaconda;

    ```bash
      % wget https://repo.anaconda.com/archive/Anaconda3-<version>-Linux-x86_64.sh
      % bash Anaconda3-<version>-Linux-x86_64.sh
    ```

1. create conda environment:

   ```bash
     % conda create -n iaf2 python=3.9 -y
     % conda activate iaf2
   ```

1. install build env:

    ```bash
      % conda install -y cmake, gcc=9.4.0, gxx, gcc_linux-64, gxx_linux-64, ninja -c conda-forge
      % # Please ensure gcc >= 9.4
    ```

1. install oneAPI HPC Toolkit latest version:

    https://www.intel.cn/content/www/cn/zh/high-performance-computing/hpc-software-and-programming.html

1. initialize oneAPI env:

    ```bash
      % source <oneapi-root>/setvars.sh
    ```

    or directly load related lib files

    ```bash
      % export LD_PRELOAD=<oneapi-root>/intelpython/python3.9/lib/libiomp5.so
      % alias icc=<oneapi-root>/compiler/<onepai-version>/linux/bin/intel64/icc
    ```

1. update submodules

    ```bash
      % git submodule update --init --recursive
    ```

1. build dependencies for preprocessing:

    build AVX512-optimized hh-suite
    ```bash
      % export IAF2_DIR=`pwd`
      % git clone --recursive https://github.com/IntelLabs/hh-suite
      % cd hh-suite
      % mkdir build && cd build
      % cmake -DCMAKE_INSTALL_PREFIX=`pwd`/release -DCMAKE_CXX_COMPILER="icc" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=icelake-server" ..
      % make && make install
      % ./release/bin/hhblits -h
      % export PATH=`pwd`/release/bin:$PATH
      % cd $IAF2_DIR
    ```

    build AVX512-optimized hmmer
    ```bash
      % export IAF2_DIR=`pwd`
      % git clone --recursive https://github.com/IntelLabs/hmmer
      % source <intel-oneapi>/tbb/latest/env/vars.sh
      % cd hmmer
      % git clone https://github.com/EddyRivasLab/easel.git
      % cd easel && make clean && autoconf && ./configure --prefix=`pwd` && cd ..
      % CC=icc CFLAGS="-O3 -march=icelake-server -fPIC" ./configure --prefix=`pwd`/release
      % make && make install
      % ./release/bin/jackhmmer -h
      % export PATH=`pwd`/release/bin:$PATH
      % cd $IAF2_DIR

1. build dependency for TPP optimization of AlphaFold2 [Global]Attention Modules:

    TPP-extension is a small-matmul based practice for memory-cache balance on Xeon CPU
    It is highly recommended to setup this even if it is optional.
    If setup failed, AlphaFold2 will fall back to enable PyTorch JIT w/o PCL-extension.
    ```bash
      % export IAF2_DIR=`pwd`
      % git clone https://github.com/libxsmm/tpp-pytorch-extension
      % cd tpp-pytorch-extension
      % git submodule update --init
      % cd libxsmm && make CC=cc && cd -
      % python setup.py install
      % python -c "from tpp_pytorch_extension.alphafold.Alpha_Attention import GatingAttentionOpti_forward"
    ```

1. Put your query sequence files in "\<input-dir\>" folder:

   all fasta sequences should be named as *.fa
   1 sequence per each file, e.g. example.fa

   ```fasta
    > example file
    ATGCCGCATGGTCGTC
   ```

1. run main scripts to test your env
    
   run one_preproc.sh to do MSA and template search on 1st sample in $root_home/samples
   ```bash
     % bash online_preproc_baremetal.sh <root_home> <data-dir> <input-dir> <output-dir>
     % # please ensure your query sequence files *.fa are in <input-dir>
   ```
   intermediates data can be seen under $root_home/experiments/<sample-name>/intermediates and $root_home/experiments/<sample-name>/msa
   
   run one_modelinfer_pytorch_jit.sh to predict unrelaxed structures from MSA and template results
   ```bash
   bash one_modelinfer_pytorch_jit.sh
   ```
   unrelaxed data can be seen under $root_home/experiments/<sample-name>
   
   run one_amber.sh to relax the predicted structures from model inference:
   ```bash
   bash one_amber.sh
   ```
   relaxed data can be seen under $root_home/experiments/<sample-name>


## All steps are ended here for optimized AlphaFold2. The following lines are stock information of Original repo:

1. Update is on schedule: AlphaFold with Multimers will be coming soon

### Genetic databases

This step requires `aria2c` to be installed on your machine.

AlphaFold needs multiple genetic (sequence) databases to run:

*   [UniRef90](https://www.uniprot.org/help/uniref),
*   [MGnify](https://www.ebi.ac.uk/metagenomics/),
*   [BFD](https://bfd.mmseqs.com/),
*   [Uniclust30](https://uniclust.mmseqs.com/),
*   [PDB70](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/),
*   [PDB](https://www.rcsb.org/) (structures in the mmCIF format).

We provide a script `scripts/download_all_data.sh` that can be used to download
and set up all of these databases:

*   Default:

    ```bash
    scripts/download_all_data.sh <DOWNLOAD_DIR>
    ```

    will download the full databases.

*   With `reduced_dbs`:

    ```bash
    scripts/download_all_data.sh <DOWNLOAD_DIR> reduced_dbs
    ```

    will download a reduced version of the databases to be used with the
    `reduced_dbs` preset.

We don't provide exactly the versions used in CASP14 -- see the [note on
reproducibility](#note-on-reproducibility). Some of the databases are mirrored
for speed, see [mirrored databases](#mirrored-databases).

**Note: The total download size for the full databases is around 415 GB
and the total size when unzipped is 2.2 TB. Please make sure you have a large
enough hard drive space, bandwidth and time to download. We recommend using an
SSD for better genetic search performance.**

This script will also download the model parameter files. Once the script has
finished, you should have the following directory structure:

```
$DOWNLOAD_DIR/                             # Total: ~ 2.2 TB (download: 438 GB)
    bfd/                                   # ~ 1.7 TB (download: 271.6 GB)
        # 6 files.
    mgnify/                                # ~ 64 GB (download: 32.9 GB)
        mgy_clusters_2018_12.fa
    params/                                # ~ 3.5 GB (download: 3.5 GB)
        # 5 CASP14 models,
        # 5 pTM models,
        # LICENSE,
        # = 11 files.
    pdb70/                                 # ~ 56 GB (download: 19.5 GB)
        # 9 files.
    pdb_mmcif/                             # ~ 206 GB (download: 46 GB)
        mmcif_files/
            # About 180,000 .cif files.
        obsolete.dat
    small_bfd/                             # ~ 17 GB (download: 9.6 GB)
        bfd-first_non_consensus_sequences.fasta
    uniclust30/                            # ~ 86 GB (download: 24.9 GB)
        uniclust30_2018_08/
            # 13 files.
    uniref90/                              # ~ 58 GB (download: 29.7 GB)
        uniref90.fasta
```

`bfd/` is only downloaded if you download the full databasees, and `small_bfd/`
is only downloaded if you download the reduced databases.

### Model parameters

While the AlphaFold code is licensed under the Apache 2.0 License, the AlphaFold
parameters are made available for non-commercial use only under the terms of the
CC BY-NC 4.0 license. Please see the [Disclaimer](#license-and-disclaimer) below
for more detail.

The AlphaFold parameters are available from
https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar, and
are downloaded as part of the `scripts/download_all_data.sh` script. This script
will download parameters for:

*   5 models which were used during CASP14, and were extensively validated for
    structure prediction quality (see Jumper et al. 2021, Suppl. Methods 1.12
    for details).
*   5 pTM models, which were fine-tuned to produce pTM (predicted TM-score) and
    predicted aligned error values alongside their structure predictions (see
    Jumper et al. 2021, Suppl. Methods 1.9.7 for details).

## Running AlphaFold

**Recommended server configuration**

1. CPU: 2-sockets, Intel® Xeon® Scalable Performance Processor (61xx, 81xx, 62xx, 82xx, 92xx, 63xx, 83xx, etc.)
2. Memory: DRAM >192GB, or Intel® Optane® Persistent Memory (PMem) for higher Memory (e.g. 6TB/socket)
3. Disk: Intel® Optane® SSD 

**We need to extract the original model parameters into directory tree, so that PyTorch version of Alphafold2 can easily load params w/o mistakes.** Please use `extract_params.py` to execute such convertion.

1. Create new repository for extracted weights

   ```bash
   mkdir ~/weights
   ```

1.  Locate the original model `params`, which is set as option `--input` of script `extract_params.py`
    
    such source parameter file can be like this: `params/params_model_1.npz` or `params/params_model_2.npz`
    
1.  Define output directory as`--output_dir` 
    the script `extract_params.py` will extract original `.npz` file into a directory tree at `--output_dir`
    
    for model_1, it can be like this: `~/weights/model_1`
    
1.  Execute:

    ```bash
    python extract_params.py --input <input-npz-file> --output_dir ~/weights/model_1
    ```

1.  Notice that, `~/weights/model_1` contains a folder tree, and its root is alphafold
    
1.  Edit `numa_n_preproc.sh` to define inputs to preprocessing pipeline of AlphaFold2
    
    ```bash
    input_dir=<path-to-fasta-files> # e.g. sample.fasta is contained in data/folder1/, then put data/folder1/ here
    out_dir=<path-to-output-data> # this destination folder will contain data files alphafold2 generates
    data_dir=<root-of-alphafold-genetic-databases> # the parent folder that contains params/, bfd/, etc.
    log_dir=~/logs # the parent folder of standard outputs for each preprocessing pipeline
    prefix="mmcif_6yke-" # your input sample prefix
    suffix=".fa" # fasta file suffix
    n_sample=$1 # index of input fasta
    n_core=28 # physical cores of your CPU (total number of 1-socket CPU)
    n_socket=2 # number of CPU sockets
    ((n_sample_0=$n_sample-1))
    ((core_per_instance=$n_core*$n_socket/$n_sample))
    script="python run_preprocess.py"
    
    for i in `seq 0 ${n_sample_0}`; do
      f="$prefix${i}$suffix"
    	((lo=$i*$core_per_instance))
    	((hi=($i+1)*$core_per_instance-1))
      ((m=$i/($n_sample/2)))
    	((ncpu=$core_per_instance))
      echo preprocessing ${input_dir}/${f} on core $lo to $hi of socket $m
    	
    	numactl -C $lo-$hi -m $m $script \
    	  --n_cpu $ncpu \
    		--fasta_paths ${input_dir}/${f} \
    		--output_dir ${out_dir} \
    		--bfd_database_path=${data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    		--model_names=model_1 \
    		--uniclust30_database_path=${data_dir}/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    		--uniref90_database_path=${data_dir}/uniref90/uniref90.fasta \
    		--mgnify_database_path=${data_dir}/mgnify/mgy_clusters.fa \
    		--pdb70_database_path=${data_dir}/pdb70/pdb70 \
    		--template_mmcif_dir=${data_dir}/pdb_mmcif/mmcif_files \
    		--data_dir=${data_dir} \
    		--max_template_date=2020-05-14 \
    		--obsolete_pdbs_path=${data_dir}/pdb_mmcif/obsolete.dat \
    		--hhblits_binary_path=`which hhblits` \
    		--hhsearch_binary_path=`which hhsearch` \
    		--jackhmmer_binary_path=`which jackhmmer` \
    		--kalign_binary_path=`which kalign` \
    		> ${log_dir}/${f}.txt 2>&1 &
    done
    
    ```
    
    By default, pre-compiled dependencies will provide fast enough packages for preprocessing;
    
    But if we re-compile these programs from sources with the following GCC configurations, it will accelerate during preprocessing. Take ICX8358 as an example:
    
    ```bash
    -O2 -O3 -no-prec-div -march=icelake-server
    ```
    
    This option will take advantage of high bandwidth on an AVX512-enabled CPU.
    
    This preprocess will generate two data files as input for model inference:
    
      `features.npz`, `processed_features.npz`
    
1.  Edit `af2pth.sh` to launch the model inference
    the parameters are similar to step 6, with the following exceptions:
    
    ```bash
    input_dir=<path-to-samples> # root path containing fasta files
    out_dir=<path-to-samples> # [real i/o path for model infer] containing intermediates/ subfolder (which includes 2 npz files)
    data_dir=<root-of-alphafold-genetic-databases> # the parent folder that contains params/, bfd/, etc.
    log_dir=~/logs # the parent folder of standard outputs for each preprocessing pipeline
    prefix="mmcif_6yke-"
    suffix=".fa"
    n_sample=56 # no use
    script='run_modelinfer.py'
    model_name='model_1'
    root_params=~/weights/${model_name} # extracted weights preprocessed by extract_params.py
    
    for i in 0; do
      f="$prefix${i}$suffix"
      echo modelinfer ${input_dir}/${f}
    	python $script \
    	  --n_cpu 16 \
    		--fasta_paths ${input_dir}/${f} \
    		--output_dir ${out_dir} \
    		--bfd_database_path=${data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    		--model_names=${model_name} \
    		--root_params=${root_params} \
    		--uniclust30_database_path=${data_dir}/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    		--uniref90_database_path=${data_dir}/uniref90/uniref90.fasta \
    		--mgnify_database_path=${data_dir}/mgnify/mgy_clusters.fa \
    		--pdb70_database_path=${data_dir}/pdb70/pdb70 \
    		--template_mmcif_dir=${data_dir}/pdb_mmcif/mmcif_files \
    		--data_dir=${data_dir} \
    		--max_template_date=2020-05-14 \
    		--obsolete_pdbs_path=${data_dir}/pdb_mmcif/obsolete.dat \
    		--hhblits_binary_path=`which hhblits` \
    		--hhsearch_binary_path=`which hhsearch` \
    		--jackhmmer_binary_path=`which jackhmmer` \
    		--kalign_binary_path=`which kalign` \
    		#> ${log_dir}/${f}_${model_name}.txt
    done
    
    ```

### AlphaFold output

The outputs will be in a subfolder of `output_dir` in `run_docker.py`. They
include the computed MSAs, unrelaxed structures, relaxed structures, ranked
structures, raw model outputs, prediction metadata, and section timings. The
`output_dir` directory will have the following structure:

```
<target_name>/
    features.pkl
    ranked_{0,1,2,3,4}.pdb
    ranking_debug.json
    relaxed_model_{1,2,3,4,5}.pdb
    result_model_{1,2,3,4,5}.pkl
    timings.json
    unrelaxed_model_{1,2,3,4,5}.pdb
    msas/
        bfd_uniclust_hits.a3m
        mgnify_hits.sto
        uniref90_hits.sto
    intermediates/
        features.npz
        processed_features.npz
```

The contents of each output file are as follows:

*   `features.pkl` – A `pickle` file containing the input feature NumPy arrays
    used by the models to produce the structures.
*   `unrelaxed_model_*.pdb` – A PDB format text file containing the predicted
    structure, exactly as outputted by the model.
*   `relaxed_model_*.pdb` – A PDB format text file containing the predicted
    structure, after performing an Amber relaxation procedure on the unrelaxed
    structure prediction (see Jumper et al. 2021, Suppl. Methods 1.8.6 for
    details).
*   `ranked_*.pdb` – A PDB format text file containing the relaxed predicted
    structures, after reordering by model confidence. Here `ranked_0.pdb` should
    contain the prediction with the highest confidence, and `ranked_4.pdb` the
    prediction with the lowest confidence. To rank model confidence, we use
    predicted LDDT (pLDDT) scores (see Jumper et al. 2021, Suppl. Methods 1.9.6
    for details).
*   `ranking_debug.json` – A JSON format text file containing the pLDDT values
    used to perform the model ranking, and a mapping back to the original model
    names.
*   `timings.json` – A JSON format text file containing the times taken to run
    each section of the AlphaFold pipeline.
*   `msas/` - A directory containing the files describing the various genetic
    tool hits that were used to construct the input MSA.
*   `result_model_*.pkl` – A `pickle` file containing a nested dictionary of the
    various NumPy arrays directly produced by the model. In addition to the
    output of the structure module, this includes auxiliary outputs such as:

    *   Distograms (`distogram/logits` contains a NumPy array of shape [N_res,
        N_res, N_bins] and `distogram/bin_edges` contains the definition of the
        bins).
    *   Per-residue pLDDT scores (`plddt` contains a NumPy array of shape
        [N_res] with the range of possible values from `0` to `100`, where `100`
        means most confident). This can serve to identify sequence regions
        predicted with high confidence or as an overall per-target confidence
        score when averaged across residues.
    *   Present only if using pTM models: predicted TM-score (`ptm` field
        contains a scalar). As a predictor of a global superposition metric,
        this score is designed to also assess whether the model is confident in
        the overall domain packing.
    *   Present only if using pTM models: predicted pairwise aligned errors
        (`predicted_aligned_error` contains a NumPy array of shape [N_res,
        N_res] with the range of possible values from `0` to
        `max_predicted_aligned_error`, where `0` means most confident). This can
        serve for a visualisation of domain packing confidence within the
        structure.

The pLDDT confidence measure is stored in the B-factor field of the output PDB
files (although unlike a B-factor, higher pLDDT is better, so care must be taken
when using for tasks such as molecular replacement).

This code has been tested to match mean top-1 accuracy on a CASP14 test set with
pLDDT ranking over 5 model predictions (some CASP targets were run with earlier
versions of AlphaFold and some had manual interventions; see our forthcoming
publication for details). Some targets such as T1064 may also have high
individual run variance over random seeds.



## Citing this work

If you use the code or data in this package, please cite:

```bibtex
@Article{AlphaFold2021,
  author  = {Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and {\v{Z}}{\'\i}dek, Augustin and Potapenko, Anna and Bridgland, Alex and Meyer, Clemens and Kohl, Simon A A and Ballard, Andrew J and Cowie, Andrew and Romera-Paredes, Bernardino and Nikolov, Stanislav and Jain, Rishub and Adler, Jonas and Back, Trevor and Petersen, Stig and Reiman, David and Clancy, Ellen and Zielinski, Michal and Steinegger, Martin and Pacholska, Michalina and Berghammer, Tamas and Bodenstein, Sebastian and Silver, David and Vinyals, Oriol and Senior, Andrew W and Kavukcuoglu, Koray and Kohli, Pushmeet and Hassabis, Demis},
  journal = {Nature},
  title   = {Highly accurate protein structure prediction with {AlphaFold}},
  year    = {2021},
  doi     = {10.1038/s41586-021-03819-2},
  note    = {(Accelerated article preview)},
}
```

## Community contributions

Colab notebooks provided by the community (please note that these notebooks may
vary from our full AlphaFold system and we did not validate their accuracy):

*   The [ColabFold AlphaFold2 notebook](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb)
    by Martin Steinegger, Sergey Ovchinnikov and Milot Mirdita, which uses an
    API hosted at the Södinglab based on the MMseqs2 server [(Mirdita et al.
    2019, Bioinformatics)](https://academic.oup.com/bioinformatics/article/35/16/2856/5280135)
    for the multiple sequence alignment creation.

## Acknowledgements

AlphaFold communicates with and/or references the following separate libraries
and packages:

*   [Abseil](https://github.com/abseil/abseil-py)
*   [Biopython](https://biopython.org)
*   [Chex](https://github.com/deepmind/chex)
*   [Colab](https://research.google.com/colaboratory/)
*   [Docker](https://www.docker.com)
*   [HH Suite](https://github.com/soedinglab/hh-suite)
*   [HMMER Suite](http://eddylab.org/software/hmmer)
*   [Haiku](https://github.com/deepmind/dm-haiku)
*   [Immutabledict](https://github.com/corenting/immutabledict)
*   [JAX](https://github.com/google/jax/)
*   [Kalign](https://msa.sbc.su.se/cgi-bin/msa.cgi)
*   [matplotlib](https://matplotlib.org/)
*   [ML Collections](https://github.com/google/ml_collections)
*   [NumPy](https://numpy.org)
*   [OpenMM](https://github.com/openmm/openmm)
*   [OpenStructure](https://openstructure.org)
*   [pymol3d](https://github.com/avirshup/py3dmol)
*   [SciPy](https://scipy.org)
*   [Sonnet](https://github.com/deepmind/sonnet)
*   [TensorFlow](https://github.com/tensorflow/tensorflow)
*   [Tree](https://github.com/deepmind/tree)
*   [tqdm](https://github.com/tqdm/tqdm)

We thank all their contributors and maintainers!

## License and Disclaimer

This is not an officially supported Google product.

Copyright 2021 DeepMind Technologies Limited.

### AlphaFold Code License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

### Model Parameters License

The AlphaFold parameters are made available for non-commercial use only, under
the terms of the Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0) license. You can find details at:
https://creativecommons.org/licenses/by-nc/4.0/legalcode

### Third-party software

Use of the third-party software, libraries or code referred to in the
[Acknowledgements](#acknowledgements) section above may be governed by separate
terms and conditions or license provisions. Your use of the third-party
software, libraries or code is subject to any such terms and you should check
that you can comply with any applicable restrictions or terms and conditions
before use.

### Mirrored Databases

The following databases have been mirrored by DeepMind, and are available with reference to the following:

*   [BFD](https://bfd.mmseqs.com/) (unmodified), by Steinegger M. and Söding J., available under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

*   [BFD](https://bfd.mmseqs.com/) (modified), by Steinegger M. and Söding J., modified by DeepMind, available under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/). See the Methods section of the [AlphaFold proteome paper](https://www.nature.com/articles/s41586-021-03828-1) for details.

*   [Uniclust30: v2018_08](http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/) (unmodified), by Mirdita M. et al., available under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

*   [MGnify: v2018_12](http://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/current_release/README.txt) (unmodified), by Mitchell AL et al., available free of all copyright restrictions and made fully and freely available for both non-commercial and commercial use under [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).

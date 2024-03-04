# AlphaFold2 optimized on Intel Xeon CPU

Key words:
  Intel AlphaFold2, Open-omics-alphafold, AlphaFold2 on CPU, AlphaFold2 on Xeon, AlphaFold2 inference on SPR AVX512 FP32 and AMX-BF16

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


## Primary solution for setup of Open-omics-alphafold environment

1. install anaconda;

    ```bash
      wget https://repo.anaconda.com/archive/Anaconda3-<version>-Linux-x86_64.sh
      bash Anaconda3-<version>-Linux-x86_64.sh
    ```

2. create conda environment using a .yml file:

   ```bash
     conda env create -f conda_requirements.yml
     conda activate iaf2
   ```


3. install oneAPI Base Toolkit and oneAPI HPC Toolkit latest version:

    https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2023-2/overview.html

4. initialize oneAPI env:

    ```bash
      source <oneapi-root>/setvars.sh            # reactivate the conda environment of previous step after sourcing (conda activate iaf2)
    ```

    or directly source compiler and mkl

    ```bash
      source /opt/intel/oneapi/compiler/latest/env/vars.sh intel64
      source /opt/intel/oneapi/mkl/latest/env/vars.sh intel64
    ```

    (Optional) set library path if needed
    ```bash
    export LD_PRELOAD=/opt/intel/oneapi/intelpython/python3.9/lib/libiomp5.so:$LD_PRELOAD
    ```

5. update submodules

    ```bash
      git submodule update --init --recursive
    ```

6. Build dependencies for preprocessing (Optimized hh-suite and hmmer):

    (GCC >= 9.4.0 and cmake is required)
    build AVX512-optimized hh-suite
    ```bash
      export IAF2_DIR=`pwd`
      git clone --recursive https://github.com/IntelLabs/hh-suite.git
      cd hh-suite
      mkdir build && cd build
      cmake -DCMAKE_INSTALL_PREFIX=`pwd`/release -DCMAKE_CXX_COMPILER="icpx" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native" ..
      make -j 4 && make install
      ./release/bin/hhblits -h
      export PATH=`pwd`/release/bin:$PATH
      cd $IAF2_DIR
    ```

    build AVX512-optimized hmmer
    ```bash
      export IAF2_DIR=`pwd`
      git clone --recursive https://github.com/IntelLabs/hmmer.git
      source <intel-oneapi>/tbb/latest/env/vars.sh
      cd hmmer
      cd easel && make clean && autoconf && ./configure --prefix=`pwd` && cd ..
      autoconf && CC=icx CFLAGS="-O3 -march=native -fPIC" ./configure --prefix=`pwd`/release
      make -j 4 && make install
      ./release/bin/jackhmmer -h
      export PATH=`pwd`/release/bin:$PATH
      cd $IAF2_DIR
    ```

7. build dependency for TPP optimization of AlphaFold2 [Global]Attention Modules:

    TPP-pytorch-extension implements efficient kernels for Xeon CPUs in C++ using the libxsmm library.
    If setup failed, AlphaFold2 will fall back to enable PyTorch JIT w/o PCL-extension.
    ```bash
    export IAF2_DIR=`pwd`
    git clone https://github.com/libxsmm/tpp-pytorch-extension
    cd tpp-pytorch-extension
    git submodule update --init
    python setup.py install
    python -c "from tpp_pytorch_extension.alphafold.Alpha_Attention import GatingAttentionOpti_forward"
    ```
8. extract weights in the <root_home> directory

    ```bash
    mkdir weights && mkdir weights/extracted
    python extract_params.py --input <data-dir>/params/params_model_1.npz --output_dir ./weights/extracted/model_1
    ```
   
9. Put your query sequence files in "\<input-dir\>" folder:

   all fasta sequences should be named as *.fa
   1 sequence per each file, e.g. example.fa

   ```fasta
    > example file
    ATGCCGCATGGTCGTC
   ```

10. run main scripts to test your env
    
   run preprocess main script to do MSA and template search on 1st sample in $root_home/samples
   ```bash
     bash online_preproc_baremetal.sh <root_home> <data-dir> <input-dir> <output-dir>
     # please ensure your query sequence files *.fa are in <input-dir>
   ```
   intermediates data can be seen under $output-dir/<sample-name>/intermediates and $output-dir/<sample-name>/msa
   
   run model inference script to predict unrelaxed structures from MSA and template results
   ```bash
   bash online_inference_baremetal.sh <conda_env_path> <root_home> <data-dir> <input-dir> <output-dir> <model_name>
   ```
   unrelaxed data can be seen under $output-dir/<sample-name>

11. Run relaxation script (Untested)

    Download stereo_chemical_props.txt file into alphafold/common folder using the following command
    ```bash
    wget -q -P ./alphafold/common/ https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt --no-check-certificate
    ```

    Run the relaxation script with the following command
    ```bash
    bash one_amber.sh <conda_env_path> <root_home> <data-dir> <input-dir> <output-dir> <model_name>
    ```

12. Multi-instance Throughput Run 
    First, create a logs directory in the <root_home> directory with the following command 
    ```bash
    mkdir <root_home>/logs
    ```
    Run the multi-instance preprocessing script with the following command
    ```bash
    python run_multiprocess_pre.py --root_home=<root_home> --data_dir=<data_dir> --input_dir=<input_dir> --output_dir=<output_dir> --model_name=<model_name>

    ```
    Run the multi-instance model inference script with the following command
    ```bash
    python run_multiprocess_infer.py --root_condaenv=<conda_env_path> --root_home=<root_home> --data_dir=<data_dir> --input_dir=<input_dir> --output_dir=<output_dir> --model_name=<model_name>
    ```

## All steps are ended here for optimized AlphaFold2. 
## The following lines are stock information of the Original Alphafold2 repo:

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

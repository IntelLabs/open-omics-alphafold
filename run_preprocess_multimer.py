# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AlphaFold protein structure prediction script."""
# import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Dict, Union
from runners.timmer import Timmers
from runners.saver import save_feature_dict, load_feature_dict_if_exist

from absl import app
from absl import flags
from absl import logging
from datapipeline_parallel import DataPipeline
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch

# Internal import (7716).

logging.set_verbosity(logging.INFO)

flags.DEFINE_list(
  'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
  'target that will be folded one after another. If a FASTA file contains '
  'multiple sequences, then it will be folded as a multimer. Paths should be '
  'separated by commas. All FASTA paths must have a unique basename as the '
  'basename is used to name the output directories for each prediction.')

flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('jackhmmer_binary_path', shutil.which('jackhmmer'),
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', shutil.which('hhblits'),
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', shutil.which('hhsearch'),
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', shutil.which('hmmsearch'),
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', shutil.which('hmmbuild'),
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('kalign_binary_path', shutil.which('kalign'),
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniref30_database_path', None, 'Path to the UniRef30 '
                    'database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', None, 'Path to the Uniprot '
                    'database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', None, 'Path to the PDB '
                    'seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_enum('db_preset', 'full_dbs',
                  ['full_dbs', 'reduced_dbs'],
                  'Choose preset MSA database configuration - '
                  'smaller genetic database config (reduced_dbs) or '
                  'full genetic database config  (full_dbs)')
flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('num_multimer_predictions_per_model', 5, 'How many '
                     'predictions (each with a different random seed) will be '
                     'generated per model. E.g. if this is 2 and there are 5 '
                     'models then there will be 10 predictions per input. '
                     'Note: this FLAG only applies if model_preset=multimer')
flags.DEFINE_boolean('use_precomputed_msas', False, 'Whether to read MSAs that '
                     'have been written to disk instead of running the MSA '
                     'tools. The MSA files are looked up in the output '
                     'directory, so it must stay the same between multiple '
                     'runs that are to reuse the MSAs. WARNING: This will not '
                     'check if the sequence, database or configuration have '
                     'changed.')
flags.DEFINE_boolean('run_relax', True, 'Whether to run the final relaxation '
                     'step on the predicted models. Turning relax off might '
                     'result in predictions with distracting stereochemical '
                     'violations but might help in case you are having issues '
                     'with the relaxation stage.')
flags.DEFINE_boolean('run_in_parallel', False, 'Whether to run the MSA in parallel ')
flags.DEFINE_boolean('use_gpu_relax', None, 'Whether to relax on GPU. '
                     'Relax on GPU can be much faster than CPU, so it is '
                     'recommended to enable if possible. GPUs must be available'
                     ' if this setting is enabled.')
flags.DEFINE_integer('n_cpu', None, 'CPU physical cores used in MSA '
                    'It is dependent on the instance number you want to run '
                    'simultaneosly. e.g. your #CPU_core=32 & #instance=8, '
                    'choose 4', lower_bound=1, required=True)

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')


def predict_structure(
    timmer: Timmers,
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    model_runners: Dict[str, model.RunModel],
    random_seed: int):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  tmp_output_dir = os.path.join(output_dir, 'intermediates')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)
  if not os.path.exists(tmp_output_dir):
    os.makedirs(tmp_output_dir)

  is_save_intermediates = True
  # Get features.
  t_0 = time.time()
  timmer.add_timmer('predict_%s_datapipeline' % fasta_name)
  ftmp_featdict = os.path.join(tmp_output_dir, 'features.npz')
  feature_dict = load_feature_dict_if_exist(ftmp_featdict)
  if feature_dict is None:
    print('#### 1. start data pipeline preprocessing from de novo.')
    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir)
    if is_save_intermediates:
      save_feature_dict(ftmp_featdict, feature_dict)
  else:
    print('==== 1. loaded archive of data pipeline preprocessing.')
  timings['features'] = time.time() - t_0
  timmer.end_timmer('predict_%s_datapipeline' % fasta_name)
  timmer.save()

  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  with open(features_output_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)

  unrelaxed_pdbs = {}
  relax_metrics = {}
  ranking_confidences = {}

  # Run the models.
  num_models = len(model_runners)
  for model_index, (model_name, model_runner) in enumerate(
      model_runners.items()):
    logging.info('Running model %s on %s', model_name, fasta_name)
    t_0 = time.time()
    timmer.add_timmer('processfeatures_%s_by_%s' % (fasta_name, model_name))
    ftmp_processed_featdict = os.path.join(tmp_output_dir, 'processed_features.npz')
    processed_feature_dict = load_feature_dict_if_exist(ftmp_processed_featdict)
    logging.info('#### 2. start feature pre-model processing from de novo.')
    model_random_seed = model_index + random_seed * num_models
    processed_feature_dict = model_runner.process_features(
      feature_dict, 
      random_seed=model_random_seed
    )
    if is_save_intermediates:
      save_feature_dict(ftmp_processed_featdict, processed_feature_dict)      

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for tool_name in (
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
    if not FLAGS[f'{tool_name}_binary_path'].value:
      raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                       'sure it is installed on your system.')

  use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
  _check_flag('small_bfd_database_path', 'db_preset',
              should_be_set=use_small_bfd)
  _check_flag('bfd_database_path', 'db_preset',
              should_be_set=not use_small_bfd)
  _check_flag('uniref30_database_path', 'db_preset',
              should_be_set=not use_small_bfd)

  run_multimer_system = 'multimer' in FLAGS.model_preset
  _check_flag('pdb70_database_path', 'model_preset',
              should_be_set=not run_multimer_system)
  _check_flag('pdb_seqres_database_path', 'model_preset',
              should_be_set=run_multimer_system)
  _check_flag('uniprot_database_path', 'model_preset',
              should_be_set=run_multimer_system)

  if FLAGS.model_preset == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  # init timmers
  f_timmer = os.path.join(FLAGS.output_dir, 'timmers_%s.txt' % fasta_names[0])
  h_timmer = Timmers(f_timmer)

  print('### use %d CPU cores' % FLAGS.n_cpu)
  h_timmer.add_timmer('template_hit_featurizer')
  if run_multimer_system:
    print('######### multimer!!!!')
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        database_path=FLAGS.pdb_seqres_database_path)
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
  else:
    template_searcher = hhsearch.HHSearch(
        binary_path=FLAGS.hhsearch_binary_path,
        databases=[FLAGS.pdb70_database_path])
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

  h_timmer.end_timmer('template_hit_featurizer')
  h_timmer.save()
  h_timmer.add_timmer('data_pipeline')
  monomer_data_pipeline = DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniref30_database_path=FLAGS.uniref30_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=use_small_bfd,
      use_precomputed_msas=FLAGS.use_precomputed_msas,
      n_cpu=FLAGS.n_cpu,
      run_in_parallel=FLAGS.run_in_parallel)


  if run_multimer_system:
    num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        uniprot_database_path=FLAGS.uniprot_database_path,
        use_precomputed_msas=FLAGS.use_precomputed_msas)
  else:
    num_predictions_per_model = 1
    data_pipeline = monomer_data_pipeline
  h_timmer.end_timmer('data_pipeline')
  h_timmer.save()

  model_runners = {}
  model_names = config.MODEL_PRESETS[FLAGS.model_preset]
  for model_name in model_names:
    h_timmer.add_timmer('model_%s_compilation' % model_name)
    model_config = config.model_config(model_name)
    if run_multimer_system:
      model_config.model.num_ensemble_eval = num_ensemble
    else:
      model_config.data.eval.num_ensemble = num_ensemble
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = model.RunModel(model_config, model_params)
    for i in range(num_predictions_per_model):
      model_runners[f'{model_name}_pred_{i}'] = model_runner
    h_timmer.end_timmer('model_%s_compilation' % model_name)
  h_timmer.save()

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  random_seed = 5582232524994481130
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // len(model_runners))
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Predict structure for each of the sequences.
  for i, fasta_path in enumerate(FLAGS.fasta_paths):
    fasta_name = fasta_names[i]
    h_timmer.add_timmer('predict_%s' % fasta_name)
    predict_structure(
        timmer=h_timmer,
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        random_seed=random_seed)
    h_timmer.end_timmer('predict_%s' % fasta_name)
    h_timmer.save()


if __name__ == '__main__':
  flags.mark_flags_as_required([
    'fasta_paths',
    'output_dir',
    'data_dir',
    'uniref90_database_path',
    'mgnify_database_path',
    'template_mmcif_dir',
    'max_template_date',
    'obsolete_pdbs_path',
    'n_cpu',
  ])

  t1 = time.time()
  app.run(main)
  t2 = time.time()
  print('### total time: %d sec' % (t2-t1))

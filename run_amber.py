import os
import pathlib
import pickle
import time
from runners.timmer import Timmers
from runners.saver import load_feature_dict_if_exist
from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.relax import relax
import numpy as np
import jax

### Define Flags
flags.DEFINE_list('fasta_paths', None, 'Paths to FASTA files, each containing '
                  'one sequence. Paths should be separated by commas. '
                  'All FASTA paths must have a unique basename as the '
                  'basename is used to name the output directories for '
                  'each prediction.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('model_names', None, 'Names of models to use.')
flags.DEFINE_string('root_params', None, 'root directory of model parameters') ### updated
flags.DEFINE_integer('random_seed', 123, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
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
MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20


### helper func: validate required options
def _check_flag(flag_name: str, preset: str, should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set for preset "{preset}"')


### main func for model inference
def amber_relax(
    timmer: Timmers,
    fasta_name: str,
    output_dir_base: str,
    amber_relaxer: relax.AmberRelaxation):
  print('### Validate preprocessed results.')
  timings = {}
  t0_total = time.time()
  output_dir = os.path.join(output_dir_base, fasta_name)
  assert os.path.isdir(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  tmp_output_dir = os.path.join(output_dir, 'intermediates')
  assert os.path.isdir(msa_output_dir)
  assert os.path.isdir(tmp_output_dir)
  ftmp_processed_featdict = os.path.join(
    tmp_output_dir, 
    'processed_features.npz')
  processed_feature_dict = load_feature_dict_if_exist(
    ftmp_processed_featdict)
  processed_feature_dict = jax.tree_map(
    lambda x:np.array(x), processed_feature_dict)
  if processed_feature_dict is None:
    raise FileNotFoundError(
      'Invalid processed features: ',
      ftmp_processed_featdict)
  
  # model_name = FLAGS.model_names[0]
  model_list = FLAGS.model_names.strip('[]').split(',')
  num_prediction_per_model = FLAGS.num_multimer_predictions_per_model
  print(model_list)
  for model_name in model_list:
    for i in range(num_prediction_per_model):
      result_output_path = os.path.join(output_dir, f'result_{model_name}_pred_{i}.pkl')
      with open(result_output_path, 'rb') as f:
        prediction_result = pickle.load(f)
      prediction_result = jax.tree_map(
        lambda x:np.array(x), prediction_result)

      print('### load unrelaxed structure')
      if FLAGS.model_preset == 'multimer':
        unrelaxed_protein = protein.from_prediction(
          processed_feature_dict,
          prediction_result,
          remove_leading_feature_dimension=False)
      else:
        unrelaxed_protein = protein.from_prediction(
          processed_feature_dict,
          prediction_result,
          remove_leading_feature_dimension=True)

      print('### post-adjust: amber-relax')
      relaxed_pdbs = {}
      t_0 = time.time()
      timmer_name = 'amberrelax_%s_from_%s_pred_%s' % (fasta_name, model_name, str(i))
      timmer.add_timmer(timmer_name)
      t1_amber = time.time()
      relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
      t2_amber = time.time()
      print('  # [TIME] amber process =', (t2_amber-t1_amber),'sec')
      relaxed_pdbs[model_name] = relaxed_pdb_str
      f_relaxed_output = os.path.join(output_dir, f'relaxed_{model_name}_pred_{i}.pdb')
      with open(f_relaxed_output, 'w') as h:
        h.write(relaxed_pdb_str)
      timings[f'relax_{model_name}'] = time.time() - t_0
      timmer.end_timmer(timmer_name)
      timmer.save()
  t_diff = time.time() - t0_total
  timings[f'predict_and_compile_all_models'] = t_diff


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many cml args.')
  
  print('### start script for model infer.')
  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')
  # init timmers
  f_timmer = os.path.join(FLAGS.output_dir, 'timmers_%s.txt' % fasta_names[0])
  h_timmer = Timmers(f_timmer)
  # init amber
  h_timmer.add_timmer('amber_relaxation')
  amber_relaxer = relax.AmberRelaxation(
    max_iterations=RELAX_MAX_ITERATIONS,
    tolerance=RELAX_ENERGY_TOLERANCE,
    stiffness=RELAX_STIFFNESS,
    exclude_residues=RELAX_EXCLUDE_RESIDUES,
    max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
    use_gpu=False)
  h_timmer.end_timmer('amber_relaxation')
  h_timmer.save()
  # init randomizer
  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = 5582232524994481130
  logging.info('Using random seed %d for the data pipeline', random_seed)
  ### predict
  for fasta_path, fasta_name in zip(FLAGS.fasta_paths, fasta_names):
    h_timmer.add_timmer('predict_%s' % fasta_name)
    amber_relax(
      timmer=h_timmer,
      fasta_name=fasta_name,
      output_dir_base=FLAGS.output_dir,
      amber_relaxer=amber_relaxer)
    h_timmer.end_timmer('predict_%s' % fasta_name)
    h_timmer.save()


if __name__ == '__main__':
  flags.mark_flags_as_required([
    'fasta_paths',
    'output_dir',
    'model_names',
    'model_preset',
    # 'root_params',
  ])
  app.run(main)

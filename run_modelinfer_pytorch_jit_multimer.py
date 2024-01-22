from absl import logging
from absl import flags
from absl import app
import pathlib
from typing import Dict
import numpy as np
import pickle
import os
from time import time
from alphafold.common.residue_constants import atom_type_num
from alphafold_pytorch_jit.net import RunModel
from alphafold.common import protein
from alphafold.model import config
from runners.saver import load_feature_dict_if_exist
from runners.timmer import Timmers
import jax
from pdb import set_trace


logging.set_verbosity(logging.INFO)
flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, suffix should be .fa. '
    'each containing a prediction target that will be folded one after another. '
    'If a FASTA file contains multiple sequences, then it will be folded as a multimer. '
    'Paths should be separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'read inputs and store results.')
flags.DEFINE_string('model_names', None, 'names of multimer model to use')
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
FLAGS = flags.FLAGS


from alphafold_pytorch_jit.basics import GatingAttention
from tpp_pytorch_extension.alphafold.Alpha_Attention import GatingAttentionOpti_forward
GatingAttention.forward = GatingAttentionOpti_forward


def run_model_inference(
  fasta_name:str,
  output_dir_base:str,
  model_runners: Dict[str, RunModel]
):
  fasta_name = fasta_name.rstrip('.fa').rstrip('.fasta')
  logging.info('run model prediction of {}'.format(fasta_name))
  output_dir = os.path.join(output_dir_base, fasta_name)
  assert os.path.exists(output_dir)
  preproc_dir = os.path.join(output_dir, 'intermediates')
  fp_features = os.path.join(preproc_dir, 'features.npz')
  assert os.path.isdir(preproc_dir) and os.path.isfile(fp_features)
  
  # load features
  df_features = load_feature_dict_if_exist(fp_features)

  # run model inference
  durations = {}
  ranking_confidences = {}
  unrelaxed_proteins = {}
  unrelaxed_pdbs = {}
  for model_idx, (model_name, model_runner) in enumerate(
    model_runners.items()):
    logging.info('use model {} on {}'.format(
      model_name, fasta_name))
    t0 = time()
    prediction_result = model_runner(df_features)
    dt = time() - t0
    set_trace()
    durations['predict_and_compile_{}'.format(model_name)] = dt
    logging.info('complete model {} inference with duration = {}'.format(
      model_name, dt))
    plddt = prediction_result['plddt']
    ranking_confidences[model_name] = prediction_result['ranking_confidence']
    fp_output = os.path.join(output_dir, f'result_{model_name}.pkl')
    with open(fp_output, 'wb') as h:
      pickle.dump(prediction_result, h, protocol=4)
    plddt_b_factors = np.repeat(
      plddt[:, None], atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
      df_features, 
      prediction_result,
      plddt_b_factors,
      remove_leading_feature_dimension=False)
    unrelaxed_proteins[model_name] = unrelaxed_protein
    unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)

  # sort by plddt ranks
  ranked_order = [
    model_name for model_name, confidence in 
    sorted(ranking_confidences.items(), key=lambda x:x[1], reverse=True)]
  for idx, model_name in enumerate(ranked_order):
    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}_rank{idx}.pdb')
    with open(unrelaxed_pdb_path, 'w') as h:
      h.write(unrelaxed_pdbs[model_name]) # save unrelaxed pdb
  

def main(argv):
  num_ensemble = 1
  num_prediction_per_model = FLAGS.num_multimer_predictions_per_model
  model_names = FLAGS.model_names # config.MODEL_PRESETS['multimer']
  root_params = FLAGS.root_params
  if isinstance(model_names, str):
    model_names = [model_names]
  
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')
  
  model_runners = {}
  for model_name in model_names:
    model_config = config.model_config(model_name)
    model_config['model']['num_ensemble_eval'] = num_ensemble
    fp_timmer = os.path.join(FLAGS.output_dir, f'timmers_{model_name}.txt')
    h_timmer = Timmers(fp_timmer)
    for i in range(num_prediction_per_model):
      model_runners[f'{model_name}_pred_{i}'] = RunModel(
        model_config, root_params, h_timmer, FLAGS.random_seed)
  for fasta_name in fasta_names:
    run_model_inference(
      fasta_name,
      FLAGS.output_dir,
      model_runners)


if __name__ == '__main__':
  flags.mark_flags_as_required([
    'fasta_paths',
    'data_dir',
    'output_dir',
    'root_params',
    'model_names'
  ])
  app.run(main)

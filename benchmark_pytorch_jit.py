import json
from runners.timmer import Timmers
from runners.saver import load_feature_dict_if_exist

from absl import app
from absl import logging
import torch
from alphafold_haiku.common import protein, confidence
from alphafold_haiku.model import config
from alphafold_pytorch_jit import net as model
import numpy as np
import jax
import intel_extension_for_pytorch as ipex

from alphafold_pytorch_jit.basics import GatingAttention
from pcl_pytorch_extension.alphafold.Alpha_Attention import GatingAttentionOpti_forward
GatingAttention.forward = GatingAttentionOpti_forward


MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20


### main func for model inference
def dummy_infer(
    timmer: Timmers,
    seq_len: int,
    random_seed: int):
  print('### Validate preprocessed results.')
  timings = {}
  
  ### prepare model runners
  processed_feature_dict = {
      "aatype" : np.ones((4, seq_len), dtype="int32"), 
      "residue_index" : np.ones((4, seq_len), dtype="int32"), 
      "seq_length" : np.ones((4,), dtype="int32"), 
      "template_aatype" : np.ones((4, 4, seq_len), dtype="int32"), 
      "template_all_atom_masks" : np.ones((4, 4, seq_len, 37), dtype="float32"), 
      "template_all_atom_positions" : np.ones((4, 4, seq_len, 37, 3), dtype="float32"), 
      "template_sum_probs" : np.ones((4, 4, 1), dtype="float32"), 
      "is_distillation" : np.ones((4,), dtype="float32"), 
      "seq_mask" : np.ones((4, seq_len), dtype="float32"), 
      "msa_mask" : np.ones((4, 508, seq_len), dtype="float32"), 
      "msa_row_mask" : np.ones((4, 508), dtype="float32"), 
      "random_crop_to_size_seed" : np.ones((4, 2), dtype="int32"), 
      "template_mask" : np.ones((4, 4), dtype="float32"), 
      "template_pseudo_beta" : np.ones((4, 4, seq_len, 3), dtype="float32"), 
      "template_pseudo_beta_mask" : np.ones((4, 4, seq_len), dtype="float32"), 
      "atom14_atom_exists" : np.ones((4, seq_len, 14), dtype="float32"), 
      "residx_atom14_to_atom37" : np.ones((4, seq_len, 14), dtype="int32"), 
      "residx_atom37_to_atom14" : np.ones((4, seq_len, 37), dtype="int32"), 
      "atom37_atom_exists" : np.ones((4, seq_len, 37), dtype="float32"), 
      "extra_msa" : np.ones((4, 5120, seq_len), dtype="int32"), 
      "extra_msa_mask" : np.ones((4, 5120, seq_len), dtype="float32"), 
      "extra_msa_row_mask" : np.ones((4, 5120), dtype="float32"), 
      "bert_mask" : np.ones((4, 508, seq_len), dtype="float32"), 
      "extra_has_deletion" : np.ones((4, 5120, seq_len), dtype="float32"), 
      "extra_deletion_value" : np.ones((4, 5120, seq_len), dtype="float32"), 
      "msa_feat" : np.ones((4, 508, seq_len, 49), dtype="float32"), 
      "target_feat" : np.ones((4, seq_len, 22), dtype="float32")
  }
  processed_feature_dict = jax.tree_map(
    lambda x:torch.tensor(x), processed_feature_dict)
  num_ensemble = 1
  model_runners = {}
  model_name = 'model_1'
  model_config = config.model_config(model_name)
  model_config['data']['eval']['num_ensemble'] = num_ensemble
  model_runner = model.RunModel(
      model_config, 
      None, 
      timmer,
      random_seed)
  model_runners[model_name] = model_runner 
  model_runners[model_name].eval()
  model_runners[model_name] = ipex.optimize(model_runners[model_name])

  for model_name, model_runner in model_runners.items():
    print('### [INFO] Execute model inference')
    timmer_name = f'model inference: {model_name}'
    timmer.add_timmer(timmer_name)
    with torch.no_grad():
      _ = model_runner(processed_feature_dict)
    processed_feature_dict = jax.tree_map(
      lambda x:x.detach().numpy(),
      processed_feature_dict)
    timmer.end_timmer(timmer_name)
    timmer.save()

    with open('test_benchmark_timing.json', 'w') as h:
      h.write(json.dumps(timings, indent=4))


def main(argv):
  seqlen = 81
  f_timmer = 'timing_modelinfer.txt'
  h_timmer = Timmers(f_timmer)
  h_timmer.add_timmer('amber_relaxation')
  #amber_relaxer = relax.AmberRelaxation(
  #  max_iterations=RELAX_MAX_ITERATIONS,
  #  tolerance=RELAX_ENERGY_TOLERANCE,
  #  stiffness=RELAX_STIFFNESS,
  #  exclude_residues=RELAX_EXCLUDE_RESIDUES,
  #  max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)
  h_timmer.end_timmer('amber_relaxation')
  h_timmer.save()
  # init randomizer
  random_seed = 1234
  logging.info('Using random seed %d for the data pipeline', random_seed)
  ### predict
  h_timmer.add_timmer('predict_%s' % seqlen)
  dummy_infer(
      timmer=h_timmer,
      seq_len=seqlen,
      random_seed=random_seed)
  h_timmer.end_timmer('predict_%s' % seqlen)
  h_timmer.save()


if __name__ == '__main__':
  app.run(main)

from typing import Any, Union, Mapping
from alphafold_pytorch_jit import features
import tensorflow.compat.v1 as tf
from torch import nn
import os
import torch
import jax
import numpy as np
from alphafold.common import confidence
from alphafold_pytorch_jit import subnets, subnets_multimer
from alphafold_pytorch_jit.folding import StructureModule
from alphafold_pytorch_jit.utils import detached, unwrap_tensor
from alphafold_pytorch_jit.hk_io import get_pure_fn
from alphafold_pytorch_jit.weight_io import (
  load_npy2hk_params, 
  load_npy2pth_params)
from pdb import set_trace
from time import time

def get_confidence_metrics(
  prediction_result: Mapping[str, Any],
  multimer_mode: bool = False
) -> Mapping[str, Any]:
  """
    post-processes prediction_result to get confidence metrics
  """
  conf_metrics = {}
  conf_metrics['plddt'] = confidence.compute_plddt(
    prediction_result['predicted_lddt']['logits'])
  if 'predicted_aligned_error' in prediction_result.keys():
    conf_metrics.update(confidence.compute_predicted_aligned_error(
      prediction_result['predicted_aligned_error']['logits'],
      prediction_result['predicted_aligned_error']['breaks']
    ))
    conf_metrics['ptm'] = confidence.predicted_tm_score(
      prediction_result['predicted_aligned_error']['logits'],
      prediction_result['predicted_aligned_error']['breaks']
    )
    if multimer_mode:
      conf_metrics['iptm'] = confidence.predicted_tm_score(
        logits=prediction_result['predicted_aligned_error']['logits'],
        breaks=prediction_result['predicted_aligned_error']['breaks'],
        asym_id=prediction_result['predicted_aligned_error']['asym_id'],
        interface=True) # [TODO] new option in multimer version
      conf_metrics['ranking_confidence'] = (
        0.8 * conf_metrics['iptm'] + 0.2 * conf_metrics['ptm'])
  if not multimer_mode:
    conf_metrics['ranking_confidence'] = np.mean(
      conf_metrics['plddt'])
  return conf_metrics

def process_features(
    config,
    raw_features: Union[tf.train.Example, features.FeatureDict],
    random_seed: int) -> features.FeatureDict:
  """Processes features to prepare for feeding them into the model.
  Args:
    raw_features: The output of the data pipeline either as a dict of NumPy
      arrays or as a tf.train.Example.
    random_seed: The random seed to use when processing the features.
  Returns:
    A dict of NumPy feature arrays suitable for feeding into the model.
  """
  if config['model']['global_config']['multimer_mode']:
    return raw_features
  if isinstance(raw_features, dict):
    return features.np_example_to_features(
        np_example=raw_features,
        config=config,
        random_seed=random_seed)
  else:
    return features.tf_example_to_features(
        tf_example=raw_features,
        config=config,
        random_seed=random_seed)

class RunModel(object):
  def __init__(self, 
    config,
    root_params=None,
    timer=None,
    random_seed=123
  ) -> None:
    super().__init__()
    ### set hyper params
    mc = config['model']
    gc = mc['global_config']
    sc = mc['heads']['structure_module']
    self.timer = timer
    self.multimer_mode = gc['multimer_mode']
    ### load model params
    if root_params is not None:
      self.root_params = root_params
      root_af2iter = os.path.join(
        root_params, 'alphafold/alphafold_iteration') # validated
      root_struct = os.path.join(
        root_af2iter, 'structure_module')
      af2iter_params = load_npy2pth_params(root_af2iter)
      struct_params = load_npy2hk_params(root_struct)
    else:
      af2iter_params = None
      struct_params = None
    struct_rng = jax.random.PRNGKey(random_seed)
    ### create compatible structure module
    # time cost is low at structure-module
    # no need to cvt it to PyTorch version
    _, struct_apply = get_pure_fn(StructureModule, sc, gc)
    ### create AlphaFold instance
    #evo_init_dims = {
    #  'target_feat':batch['target_feat'].shape[-1],
    #  'msa_feat':batch['msa_feat'].shape[-1]
    #}

    if self.multimer_mode:
      print(f'****** ModelRunner: multimer mode AlphaFold')
      self.model = subnets_multimer.AlphaFold(
        mc,
        root_params,
        'alphafold'
      )
    else:
      evo_init_dims = {
        'target_feat': 22,
        'msa_feat': 49}
      self.model = subnets.AlphaFold(
        mc,
        evo_init_dims,
        af2iter_params,
        struct_apply,
        struct_params,
        struct_rng,
        'alphafold',
        timer=self.timer)
  
  def __call__(self, feat):
    timer_name = 'model_inference'
    if self.timer is not None:
      self.timer.add_timmer(timer_name)
    if isinstance(feat['seq_length'], torch.Tensor) and feat['seq_length'].dim() > 1:
      feat = jax.tree_map(unwrap_tensor, feat)
    if isinstance(feat['msa'], np.ndarray): # cvt numpy ndarray to torch.tensor
      for k in feat.keys():
        if isinstance(feat[k], np.ndarray):
          feat[k] = torch.Tensor(feat[k])
          if k in ['msa', 'aatype', 'template_aatype', 'residue_index', 'asym_id', 'entity_id', 'sym_id']:
            feat[k] = feat[k].to(torch.int64)
    t0 = time()
    result = self.model(feat)
    dt = time() - t0
    print(f'# [INFO] af2 iterations cost {dt} sec')
    set_trace()
    result = jax.tree_map(detached, result)
    if 'predicted_lddt' in result.keys():
      result.update(get_confidence_metrics(result))
    if self.timer is not None:
      self.timer.end_timmer(timer_name)
      self.timer.save()
    return result

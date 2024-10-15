from typing import Any, Union, Mapping
from alphafold_pytorch_jit import features
import tensorflow.compat.v1 as tf
from torch import nn
import os
import sys
import jax
import numpy as np
from alphafold.common import confidence
from alphafold_pytorch_jit import subnets
from alphafold_pytorch_jit.folding import StructureModule
from alphafold_pytorch_jit.utils import detached, unwrap_tensor
from alphafold_pytorch_jit.hk_io import get_pure_fn
from alphafold_pytorch_jit.weight_io import (
  load_npy2hk_params, 
  load_npy2pth_params)
#tpp-pytorch-extension/src/tpp_pytorch_extension/llm
'''
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tpp-pytorch-extension','src','tpp_pytorch_extension','llm'))

sys.path.append(module_path)

from llm_common import (
    BlockedLinear,
    BlockedLayerNorm,
    FixLinear,
    ShardLinear,
    get_rank,
    get_size,
    set_pg,
    _reorder_cache,
    block,
    compare,
    global_layer_dtype,
    get_layer_past_and_offset,
)
'''

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
    root_params,
    timer,
    random_seed
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
        root_params, 'alphafold/alphafold_iteration')
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
      self.model = subnets_multimer.AlphaFold(
        mc,
        af2iter_params,
        struct_apply,
        struct_params,
        struct_rng,
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

    #for m in self.modules():
    #    if isinstance(m, torch.nn.Linear):
    #       FixLinear(m, 32, 32, torch.float32)
  
  def __call__(self, feat):
    timer_name = 'model_inference'
    self.timer.add_timmer(timer_name)
    # [inc] unwrap batch data if data is unsuqeeze by INC
    if feat['seq_length'].dim() > 1:
      print('### [INFO] INC input detected')
      feat = jax.tree_map(unwrap_tensor, feat)
    result = self.model(feat)
    #del feat
    #result = jax.tree_map(cvt_result, result)
    result = jax.tree_map(detached, result)
    if 'predicted_lddt' in result.keys():
      result.update(get_confidence_metrics(result))
    self.timer.end_timmer(timer_name)
    self.timer.save()
    return result

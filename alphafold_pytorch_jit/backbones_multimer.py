import torch
from torch import nn
from alphafold_pytorch_jit.basics import (
  GatingAttention, 
  MSAColumnAttention, 
  MSARowAttentionWithPairBias, 
  MSAColumnGlobalAttention)
from alphafold_pytorch_jit.backbones import (
  Transition, 
  OuterProductMean, 
  TriangleAttention, 
  Transition)
import os
bf16 = (os.environ.get('AF2_BF16') == '1')


class FusedTriangleMultiplication(nn.Module):

  def __init__(self,config, global_config, act_dim):
    """Builds TriangleMultiplication module, w/ fused projection weights
    Arguments:
      act: Pair activations, shape [N_res, N_res, c_z]
      mask: Pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.
    Returns:
      Outputs, same shape/type as act.
    """
    super().__init__()
    self.c = config
    self.gc = global_config
    self.c_equation = self.c['equation']
    self.left_norm_input = nn.LayerNorm(act_dim)
    self.num_intermediate_channel = self.c['num_intermediate_channel']
    self.projection = nn.Linear(
      act_dim, 
      2* self.num_intermediate_channel)
    self.gate = nn.Linear(
      self.num_intermediate_channel,
      2* self.num_intermediate_channel)
    self.center_norm = nn.LayerNorm(self.num_intermediate_channel)
    self.output_projection = nn.Linear(self.num_intermediate_channel,act_dim)
    self.gating_linear = nn.Linear(act_dim,act_dim)

  def forward(self, act, mask):
    mask = mask[..., None]
    left_act = self.left_norm_input(act)
    proj_act = mask * self.projection(left_act)
    proj_act *= torch.sigmoid(self.gate(left_act))
    left_proj_act = proj_act[:, :, :self.num_intermediate_channel]
    right_proj_act = proj_act[:, :, self.num_intermediate_channel:]
    act = torch.einsum(self.c_equation, left_proj_act, right_proj_act)
    act = self.center_norm(act)
    act = self.output_projection(act)
    act *= torch.sigmoid(self.gating_linear(left_act))
    return act


class NoExtraEvoformerIteration(nn.Module):
  
  def __init__(self, config, global_config, is_extra_msa,a_dim, m_dim, pa_dim):
    super().__init__()
    """Builds EvoformerIteration module.

    Arguments:
      activations: Dictionary containing activations:
        * 'msa': MSA activations, shape [N_seq, N_res, c_m].
        * 'pair': pair activations, shape [N_res, N_res, c_z].
      masks: Dictionary of masks:
        * 'msa': MSA mask, shape [N_seq, N_res].
        * 'pair': pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.
      safe_key: prng.SafeKey encapsulating rng key.

    Returns:
      Outputs, same shape/type as act.
    """
    self.config = config
    self.global_config = global_config
    c = config
    gc = global_config
    self.is_extra_msa = is_extra_msa
    # a_dim = m_dim = 64, pa_dim = 128
    self.msa_row_attention_with_pair_bias = MSARowAttentionWithPairBias(c['msa_row_attention_with_pair_bias'],gc,a_dim, m_dim, pa_dim)
    ### this is real annoucement
    self.msa_column_attention = MSAColumnAttention(c['msa_column_attention'],gc,a_dim,a_dim,a_dim)
      #del self.msa_column_global_attention
    self.msa_transition = Transition(c['msa_transition'], gc,a_dim) # a_dim=64
    # a_dim = 64, pa_dim = 128
    self.outer_product_mean = OuterProductMean(c['outer_product_mean'], gc,a_dim,pa_dim)
    self.triangle_multiplication_outgoing = FusedTriangleMultiplication(c['triangle_multiplication_outgoing'], gc,pa_dim)
    self.triangle_multiplication_incoming = FusedTriangleMultiplication(c['triangle_multiplication_incoming'], gc,pa_dim)
    self.triangle_attention_starting_node = TriangleAttention(c['triangle_attention_starting_node'], gc,pa_dim)
    self.triangle_attention_ending_node = TriangleAttention(c['triangle_attention_ending_node'], gc,pa_dim)
    self.pair_transition = Transition(c['pair_transition'], gc, pa_dim) # pa_dim=128


  def forward(self, msa_act, pair_act, msa_mask, pair_mask):

    if bf16 == True:
      msa_act = msa_act.to(torch.bfloat16)
      pair_act = pair_act.to(torch.bfloat16)
      msa_mask = msa_mask.to(torch.bfloat16)
      pair_mask = pair_mask.to(torch.bfloat16)

    if self.config['outer_product_mean']['first']:
      pair_act = pair_act + self.outer_product_mean(msa_act,msa_mask)
    msa_act = msa_act + self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act=pair_act)
    msa_act = msa_act + self.msa_column_attention(msa_act,msa_mask)
    msa_act = msa_act + self.msa_transition(msa_act,msa_mask)
    if not self.config['outer_product_mean']['first']:
      pair_act = pair_act + self.outer_product_mean(msa_act,msa_mask)
    res = {'msa':msa_act}
    pair_act = pair_act + self.triangle_multiplication_outgoing(pair_act,pair_mask)
    pair_act = pair_act + self.triangle_multiplication_incoming(pair_act,pair_mask)
    pair_act = pair_act + self.triangle_attention_starting_node(pair_act,pair_mask)
    pair_act = pair_act + self.triangle_attention_ending_node(pair_act,pair_mask)
    pair_act = pair_act + self.pair_transition(pair_act,pair_mask)
    res['pair'] = pair_act
    return res


class ExtraEvoformerIteration(nn.Module):
  
  def __init__(self, config, global_config, is_extra_msa,a_dim, m_dim, pa_dim):
    super().__init__()
    """Builds EvoformerIteration module.

    Arguments:
      activations: Dictionary containing activations:
        * 'msa': MSA activations, shape [N_seq, N_res, c_m].
        * 'pair': pair activations, shape [N_res, N_res, c_z].
      masks: Dictionary of masks:
        * 'msa': MSA mask, shape [N_seq, N_res].
        * 'pair': pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.
      safe_key: prng.SafeKey encapsulating rng key.

    Returns:
      Outputs, same shape/type as act.
    """
    self.config = config
    self.global_config = global_config
    c = config
    gc = global_config
    self.is_extra_msa = is_extra_msa
    # a_dim = m_dim = 64, pa_dim = 128
    self.msa_row_attention_with_pair_bias = MSARowAttentionWithPairBias(c['msa_row_attention_with_pair_bias'],gc,a_dim, m_dim, pa_dim)
    ### this is real annoucement
    self.msa_column_global_attention = MSAColumnGlobalAttention(c['msa_column_attention'],gc,a_dim,a_dim,a_dim)
    #del self.msa_column_global_attention
    self.msa_transition = Transition(c['msa_transition'], gc,a_dim) # a_dim=64
    # a_dim = 64, pa_dim = 128
    self.outer_product_mean = OuterProductMean(c['outer_product_mean'], gc,a_dim,pa_dim)
    self.triangle_multiplication_incoming = FusedTriangleMultiplication(c['triangle_multiplication_incoming'], gc,pa_dim)
    self.triangle_multiplication_outgoing = FusedTriangleMultiplication(c['triangle_multiplication_outgoing'], gc,pa_dim)
    self.triangle_attention_starting_node = TriangleAttention(c['triangle_attention_starting_node'], gc,pa_dim)
    self.triangle_attention_ending_node = TriangleAttention(c['triangle_attention_ending_node'], gc,pa_dim)
    self.pair_transition = Transition(c['pair_transition'], gc, pa_dim) # pa_dim=128


  def forward(self, msa_act, pair_act, msa_mask, pair_mask):
    if self.config['outer_product_mean']['first']:
      pair_act = pair_act + self.outer_product_mean(msa_act,msa_mask)
    msa_act = msa_act + self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act=pair_act) # [TODO] CPU usage is low here
    msa_act = msa_act + self.msa_column_global_attention(msa_act,msa_mask)
    msa_act = msa_act + self.msa_transition(msa_act,msa_mask)

    if not self.config['outer_product_mean']['first']:
      pair_act = pair_act + self.outer_product_mean(msa_act,msa_mask)
    res = {'msa':msa_act}
    pair_act = pair_act + self.triangle_multiplication_outgoing(pair_act,pair_mask)
    pair_act = pair_act + self.triangle_multiplication_incoming(pair_act,pair_mask)
    pair_act = pair_act + self.triangle_attention_starting_node(pair_act,pair_mask)
    pair_act = pair_act + self.triangle_attention_ending_node(pair_act,pair_mask)

    pair_act = pair_act + self.pair_transition(pair_act,pair_mask)
    del pair_mask
    res['pair'] = pair_act
    del pair_act
    return res

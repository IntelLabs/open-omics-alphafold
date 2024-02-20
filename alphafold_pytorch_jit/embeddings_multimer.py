import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from alphafold.model.geometry import Vec3Array
from alphafold.model.folding_multimer import make_backbone_affine
from alphafold_pytorch_jit.backbones import TriangleAttention, Transition
from alphafold_pytorch_jit.backbones_multimer import FusedTriangleMultiplication
from alphafold_pytorch_jit.basics import pseudo_beta_fn_with_masks
from alphafold_pytorch_jit.basics import dgram_from_positions_pth


class TemplateEmbeddingIteration(nn.Module):
  """Single Iteration of Template Embedding."""

  def __init__(self, config, global_config, act_dim):
    # act_dim means total number of codon types
    super().__init__()
    self.c = config
    self.gc = global_config
    self.triangle_multiplication_outgoing = FusedTriangleMultiplication(
      self.c['triangle_multiplication_outgoing'], self.gc, act_dim)
    self.triangle_multiplication_incoming = FusedTriangleMultiplication(
      self.c['triangle_multiplication_incoming'], self.gc, act_dim)
    self.triangle_attention_starting_node = TriangleAttention(
      self.c['triangle_attention_starting_node'], self.gc, act_dim)
    self.triangle_attention_ending_node = TriangleAttention(
      self.c['triangle_attention_ending_node'], self.gc, act_dim)
    self.pair_transition = Transition(
      self.c['pair_transition'], self.gc, act_dim)

  def forward(self, 
              act:torch.Tensor, 
              pair_mask:torch.Tensor):
    """Build a single iteration of the template embedder.
    Args:
      act: [num_res, num_res, num_channel] Input pairwise activations.
      pair_mask: [num_res, num_res] padding mask.

    Returns:
      [num_res, num_res, num_channel] tensor of activations.
    """
    act = act + self.triangle_multiplication_outgoing(act, pair_mask)
    act = act + self.triangle_multiplication_incoming(act, pair_mask)
    act = act + self.triangle_attention_starting_node(act, pair_mask)
    act = act + self.triangle_attention_ending_node(act, pair_mask)
    act = act + self.pair_transition(act, pair_mask)
    return act


class SingleTemplateEmbedding(nn.Module):
  """Embed a single template."""

  def __init__(self, config, global_config, num_pair_channel=64):
    super().__init__()
    self.c = config
    self.gc = global_config
    act_dim = num_pair_channel
    dgram_features = self.c['dgram_features']
    self.max_bin = dgram_features['max_bin']
    self.min_bin = dgram_features['min_bin']
    self.num_bins = dgram_features['num_bins']
    self.template_stack = nn.ModuleList([
      TemplateEmbeddingIteration(self.c['template_pair_stack'], self.gc, act_dim) 
      for _ in range(self.c['template_pair_stack']['num_block'])
    ])
    self.output_layer_norm = nn.LayerNorm(
      normalized_shape=act_dim, elementwise_affine=True)
    self.query_embedding_norm = nn.LayerNorm(
      normalized_shape=2*act_dim, elementwise_affine=True
    )
    in_channels = [39, 1, 22, 22, 1, 1, 1, 1, 128] # identical for all 5 models 
    self.template_pair_embedding_stack = nn.ModuleList([
      nn.Linear(in_features=in_feature, out_features=act_dim)
      for in_feature in in_channels
    ]) # 9 means total number of to_concat embeds in self._construct_input 

  def _get_affine(self, 
    raw_atom_pos, 
    template_all_atom_mask, 
    template_aatype, 
    dtype
  ):
    atom_pos = Vec3Array.from_array(raw_atom_pos)
    rigid, backbone_mask = make_backbone_affine(
        atom_pos,
        template_all_atom_mask.detach().numpy(),
        template_aatype.detach().numpy())
    points = rigid.translation
    rigid_vec = rigid[:, None].inverse().apply_to_point(points)
    unit_vector = rigid_vec.normalized()
    unit_vector = [unit_vector.x, unit_vector.y, unit_vector.z]
    unit_vector = [torch.from_numpy(np.array(x)).to(dtype=dtype) for x in unit_vector]
    backbone_mask = torch.from_numpy(np.array(backbone_mask)).to(dtype=dtype)
    return unit_vector, backbone_mask

  @torch.jit.ignore
  def _construct_input(self,
    query_embedding:torch.Tensor, # Nres x Nres x self.c['num_channels']
    template_aatype:torch.Tensor, # Nres
    template_all_atom_positions:torch.Tensor, # Nres x 37 x 3
    template_all_atom_mask:torch.Tensor, # Nres x 37
    multichain_mask_2d:torch.Tensor,
    dtype:torch.dtype
  ):
    # Compute distogram feature for the template.
    template_positions, pseudo_beta_mask = pseudo_beta_fn_with_masks(
        template_aatype, template_all_atom_positions, template_all_atom_mask)
    pseudo_beta_mask_2d = (pseudo_beta_mask[:, None] *
                           pseudo_beta_mask[None, :])
    pseudo_beta_mask_2d *= multichain_mask_2d
    template_dgram = dgram_from_positions_pth(
        template_positions, self.num_bins, self.min_bin, self.max_bin)
    template_dgram *= pseudo_beta_mask_2d[..., None]
    template_dgram = template_dgram.to(dtype=dtype)
    pseudo_beta_mask_2d = pseudo_beta_mask_2d.to(dtype=dtype)
    to_concat = [template_dgram, pseudo_beta_mask_2d.unsqueeze(-1)] # 2
    aatype = F.one_hot(template_aatype.long(), 22).to(dtype=dtype) # , axis=-1
    to_concat.append(aatype[None, :, :]) # 2+1
    to_concat.append(aatype[:, None, :]) # 3+1
    # Compute a feature representing the normalized vector between each
    # backbone affine - i.e. in each residues local frame, what direction are
    # each of the other residues.
    raw_atom_pos = template_all_atom_positions
    raw_atom_pos = raw_atom_pos.to(torch.float32).detach().numpy()
    unit_vector, backbone_mask = self._get_affine(
      raw_atom_pos, template_all_atom_mask, template_aatype, dtype)
    backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
    backbone_mask_2d *= multichain_mask_2d
    unit_vector = [x * backbone_mask_2d for x in unit_vector]
    # Note that the backbone_mask takes into account C, CA and N (unlike
    # pseudo beta mask which just needs CB) so we add both masks as features.
    to_concat.extend([x.unsqueeze(-1) for x in unit_vector]) # 4+3
    to_concat.append(backbone_mask_2d.unsqueeze(-1)) # 7+1
    query_embedding = self.query_embedding_norm(query_embedding)
    # Allow the template embedder to see the query embedding.  Note this
    # contains the position relative feature, so this is how the network knows
    # which residues are next to each other.
    to_concat.append(query_embedding) # 8+1 = 9
    act = 0
    for x, template_pair_embedding in zip(
      to_concat, self.template_pair_embedding_stack):
      # [39, 1, 22, 22, 1, 1, 1, 1, 128]
      act += template_pair_embedding(x)
    return act

  def forward(self, 
    query_embedding:torch.Tensor, 
    template_aatype:torch.Tensor,
    template_all_atom_positions:torch.Tensor, 
    template_all_atom_mask:torch.Tensor,
    padding_mask_2d:torch.Tensor, 
    multichain_mask_2d:torch.Tensor
  ):
    """Build the single template embedding graph.

    Args:
      query_embedding: (num_res, num_res, num_channels) - embedding of the
        query sequence/msa.
      template_aatype: [num_res] aatype for each template.
      template_all_atom_positions: [num_res, 37, 3] atom positions for all
        templates.
      template_all_atom_mask: [num_res, 37] mask for each template.
      padding_mask_2d: Padding mask (Note: this doesn't care if a template
        exists, unlike the template_pseudo_beta_mask).
      multichain_mask_2d: A mask indicating intra-chain residue pairs, used
        to mask out between chain distances/features when templates are for
        single chains.
      is_training: Are we in training mode.
      safe_key: Random key generator.

    Returns:
      A template embedding (num_res, num_res, num_channels).
    """
    assert padding_mask_2d.dtype == query_embedding.dtype
    dtype = query_embedding.dtype

    act = self._construct_input(
      query_embedding, 
      template_aatype,
      template_all_atom_positions, 
      template_all_atom_mask,
      multichain_mask_2d, 
      dtype)

    for template_embedding_iteration in self.template_stack:
      act = template_embedding_iteration(
        act=act,
        pair_mask=padding_mask_2d)
    act = self.output_layer_norm(act)
    return act


class TemplateEmbedding(nn.Module):
  """Embed a set of templates."""

  def __init__(self, 
    config, 
    global_config, 
    num_pair_channel = 64): # c['embeddings_and_evoformer']['pair_channel']
    super().__init__()
    self.config = config
    self.global_config = global_config
    self.num_pair_channel = num_pair_channel
    self.template_embedder = SingleTemplateEmbedding(
      self.config, self.global_config, num_pair_channel)
    self.act_embeds = nn.ReLU()
    self.output_linear = nn.Linear(
      num_pair_channel, 2*num_pair_channel)

  def forward(self, 
    query_embedding, 
    template_aatype,
    template_all_atom_positions,
    template_all_atom_mask, 
    padding_mask_2d,
    multichain_mask_2d):
    num_templates = template_aatype.shape[0]
    num_res, _, query_num_channels = query_embedding.shape

    # Embed each template separately.
    summed_template_embeddings = torch.zeros(
      (num_res, num_res, self.num_pair_channel),
      dtype=query_embedding.dtype)
    for i in range(num_templates): # translate from hk.scan
      template_aatype_slice = template_aatype[i]
      template_all_atom_position = template_all_atom_positions[i]
      template_all_atom_mask_slice = template_all_atom_mask[i]
      # accumulate embeddings from each template
      summed_template_embeddings += self.template_embedder(
        query_embedding,
        template_aatype_slice,
        template_all_atom_position,
        template_all_atom_mask_slice,
        padding_mask_2d,
        multichain_mask_2d)
    # normalize & activate by RELU
    embedding = summed_template_embeddings / num_templates
    embedding = self.act_embeds(embedding)
    embedding = self.output_linear(embedding)
    return embedding

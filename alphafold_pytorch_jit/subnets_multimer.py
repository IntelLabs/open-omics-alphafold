from alphafold_pytorch_jit.backbones import (
  ExtraEvoformerIteration, 
  NoExtraEvoformerIteration)
from alphafold_pytorch_jit.embeddings_multimer import TemplateEmbedding
from alphafold_pytorch_jit.heads import (
  DistogramHead,
  PredictedAlignedErrorHead,
  MaskedMsaHead,
  PredictedLDDTHead,
  ExperimentallyResolvedHead
)
from alphafold_pytorch_jit.basics_multimer import (
  make_msa_profile,
  sample_msa,
  make_masked_msa,
  nearest_neighbor_clusters,
  create_msa_feat,
  pseudo_beta_fn,
  create_extra_msa_feature,
  TemplateEmbedding1d)
from alphafold_pytorch_jit.basics import mask_mean_simple
from alphafold_pytorch_jit.basics import dgram_from_positions_pth
from alphafold_pytorch_jit.utils import detached, list2tensor
from alphafold_pytorch_jit.weight_io import filtered_pth_params
from alphafold_pytorch_jit.residue_constants import atom_type_num, atom_order
from torch.nn import functional as F
from torch import nn
import torch
import jax
from runners.timmer import Timmers
import pickle
import os
from collections import OrderedDict


class EmbeddingsAndEvoformer(nn.Module):
  """Embeds the input data and runs Evoformer.

  Produces the MSA, single and pair representations.
  """
  def __init__(self, config, global_config, timmers:Timmers=None):
    super().__init__()
    self.c = config
    self.gc = global_config
    self.timmers = timmers
    self.preprocess_1d = nn.Linear(21, self.c['msa_channel']) # 21 -> 256
    self.preprocess_msa = nn.Linear(49, self.c['msa_channel']) # 49 -> 256
    self.left_single = nn.Linear(21, self.c['pair_channel']) # 21 -> 128
    self.right_single = nn.Linear(21, self.c['pair_channel']) # 21 -> 128
    self.prev_pos_linear = nn.Linear(15, self.c['pair_channel']) # 15 -> 128
    self.prev_msa_first_row_norm = nn.LayerNorm(self.c['msa_channel']) # 256
    self.prev_pair_norm = nn.LayerNorm(self.c['pair_channel']) # 128
    self.template_module = TemplateEmbedding(self.c['template'], self.gc)
    self.extra_msa_activations = nn.Linear(25, self.c['extra_msa_channel']) # 25 -> 64
    self.position_activations = nn.Linear(73, self.c['pair_channel']) # 73 -> 128
    self.extra_msa_stack = nn.ModuleList([
      ExtraEvoformerIteration(
        self.c['evoformer'], 
        self.gc, 
        True, 
        self.c['extra_msa_channel'],
        self.c['extra_msa_channel'],
        self.c['pair_channel'])
      for _ in range(self.c['extra_msa_stack_num_block'])
    ])
    self.evoformer_iteration = nn.ModuleList([
      NoExtraEvoformerIteration(
        self.c['evoformer'],
        self.gc,
        False,
        self.c['msa_channel'],
        self.c['msa_channel'],
        self.c['pair_channel'])
      for _ in range(self.c['evoformer_num_block'])
    ])
    self.template_embedding_1d = TemplateEmbedding1d(
      self.gc, self.c['msa_channel'])
    self.single_activations = nn.Linear(
      self.c['msa_channel'], self.c['seq_channel'])

  def _relative_encoding(self, 
    residue_index, # n_seq
    asym_id,
    entity_id,
    sym_id):
    """Add relative position encodings.
    For position (i, j), the value is (i-j) clipped to [-k, k] and one-hotted.
    When not using 'use_chain_relative' the residue indices are used as is, e.g.
    for heteromers relative positions will be computed using the positions in
    the corresponding chains.
    When using 'use_chain_relative' we add an extra bin that denotes
    'different chain'. Furthermore we also provide the relative chain index
    (i.e. sym_id) clipped and one-hotted to the network. And an extra feature
    which denotes whether they belong to the same chain type, i.e. it's 0 if
    they are in different heteromer chains and 1 otherwise.
    Args:
    Returns:
      Feature embedding using the features as described before.
    """
    c = self.c
    gc = self.gc
    rel_feats = []
    pos = residue_index
    asym_id_same = asym_id[:, None] == asym_id[None, :]
    offset = pos[:, None] - pos[None, :]
    # dtype = torch.bfloat16 if gc['bfloat16'] else torch.float32
    dtype = torch.float32
    clipped_offset = torch.clip(
      offset + c['max_relative_idx'], 0, 2 * c['max_relative_idx'])
    if c['use_chain_relative']:
      final_offset = torch.where(
        asym_id_same, 
        clipped_offset,
        (2 * c['max_relative_idx'] + 1) * torch.ones_like(clipped_offset))
      rel_pos = F.one_hot(final_offset, 2 * c['max_relative_idx'] + 2)
      rel_feats.append(rel_pos)
      entity_id_same = entity_id[:, None] == entity_id[None, :]
      rel_feats.append(entity_id_same.to(dtype=rel_pos.dtype)[..., None])
      rel_sym_id = sym_id[:, None] - sym_id[None, :]
      max_rel_chain = c['max_relative_chain']
      clipped_rel_chain = torch.clip(
        rel_sym_id + max_rel_chain, 0, 2 * max_rel_chain)
      final_rel_chain = torch.where(
        entity_id_same, 
        clipped_rel_chain,
        (2 * max_rel_chain + 1) * torch.ones_like(clipped_rel_chain))
      rel_chain = F.one_hot(final_rel_chain, 2 * c['max_relative_chain'] + 2)
      rel_feats.append(rel_chain)
    else:
      rel_pos = F.one_hot(clipped_offset, 2 * c['max_relative_idx'] + 1)
      rel_feats.append(rel_pos)
    rel_feat = torch.concat(rel_feats, dim=-1)
    rel_feat = rel_feat.to(dtype) # n_seq x n_seq x 73
    return self.position_activations(rel_feat) # n_seq x n_seq x 128
  
  def forward(self,
    msa:torch.Tensor, # Nmsa x Nseq
    msa_mask:torch.Tensor, # Nmsa x Nseq
    seq_mask:torch.Tensor, # Nseq
    aatype:torch.Tensor, # Nseq
    cluster_bias_mask:torch.Tensor, # Nmsa
    deletion_matrix:torch.Tensor, # Nmsa x Nseq
    bert_mask:torch.Tensor, # Nmsa x Nseq
    template_aatype:torch.Tensor, # 4 x Nseq
    template_all_atom_mask:torch.Tensor, # 4 x Nseq x 37
    template_all_atom_positions:torch.Tensor, # 4 x Nseq x 37 x 3
    residue_index:torch.Tensor, # Nseq
    asym_id:torch.Tensor, # Nseq
    entity_id:torch.Tensor, # Nseq
    sym_id:torch.Tensor, # Nseq
    prev_pos:torch.Tensor=None, # Nseq x 37 x 3
    prev_msa_first_row:torch.Tensor=None, # Nseq x 256
    prev_pair:torch.Tensor=None # Nseq x Nseq x 128
  ):
    c = self.c
    gc = self.gc
    # dtype = torch.bfloat16 if gc['bfloat16'] else torch.float32
    if self.timmers:
      self.timmers.add_timmer('multimer_embedding')
    dtype = torch.float32
    msa_profile = make_msa_profile(msa, msa_mask)
    # with torch.cpu.amp.autocast_mode(dtype=dtype): # utils.bfloat16_context
    target_feat = F.one_hot(aatype, 21).to(dtype=dtype)
    preprocess_1d = self.preprocess_1d(target_feat)
    (msa, 
     msa_mask, 
     cluster_bias_mask, 
     extra_msa,
     extra_msa_mask, 
     deletion_matrix, 
     bert_mask, 
     extra_deletion_matrix, 
     extra_bert_mask
    ) = sample_msa(
      msa,
      msa_mask,
      c['num_msa'],
      cluster_bias_mask,
      deletion_matrix,
      bert_mask)
    msa, bert_mask, true_msa = make_masked_msa(
      msa,
      msa_mask,
      msa_profile, 
      c['masked_msa'],
      bert_mask)
    cluster_profile, cluster_deletion_mean = nearest_neighbor_clusters(
      msa,
      msa_mask,
      deletion_matrix,
      extra_msa, # (n_msa - max_seq) x n_seq
      extra_msa_mask, # (n_msa - max_seq) x n_seq
      extra_deletion_matrix) # (n_msa - max_seq) x n_seq
    msa_feat = create_msa_feat( # n_msa x n_seq x 49
      msa, # n_msa x n_seq, int64
      deletion_matrix, # n_msa x n_seq
      cluster_deletion_mean, # n_msa x n_seq
      cluster_profile).to(dtype=dtype) # n_msa x n_seq x 23
    preprocess_msa = self.preprocess_msa(msa_feat)
    msa_activations = torch.unsqueeze(preprocess_1d, dim=0) + preprocess_msa
    left_single = self.left_single(target_feat) # n_seq x 128
    right_single = self.right_single(target_feat)
    pair_activations = left_single[:, None] + right_single[None] # n_seq x n_seq x 128
    mask_2d = seq_mask[:, None] * seq_mask[None, :] # n_seq x n_seq
    mask_2d = mask_2d.to(dtype=dtype)
    if c['recycle_pos']:
      prev_pseudo_beta = pseudo_beta_fn(aatype, prev_pos, None)
      dgram = dgram_from_positions_pth(
          prev_pseudo_beta, **self.c['prev_pos'])
      dgram = dgram.to(dtype=dtype)
      pair_activations += self.prev_pos_linear(dgram)
    if c['recycle_features']:
      prev_msa_first_row = self.prev_msa_first_row_norm(
        prev_msa_first_row).to(dtype=dtype)
      msa_activations[0] += prev_msa_first_row # at[0].add()
      pair_activations += self.prev_pair_norm(prev_pair).to(dtype=dtype)
    if c['max_relative_idx']:
      pair_activations += self._relative_encoding( # n_seq x n_seq x 128
        residue_index, # n_seq
        asym_id, # n_seq
        entity_id, # n_seq
        sym_id) # n_seq
    if c['template']['enabled']:
      # Construct a mask such that only intra-chain template features are
      # computed, since all templates are for each chain individually.
      multichain_mask = asym_id[:, None] == asym_id[None, :]
      template_act = self.template_module(
        query_embedding=pair_activations, # n_seq x n_seq x 128
        template_aatype=template_aatype, # 4 x n_seq
        template_all_atom_positions=template_all_atom_positions, # 4 x n_seq x 37 x 3
        template_all_atom_mask=template_all_atom_mask, # 4 x n_seq x 37
        padding_mask_2d=mask_2d, # n_seq x n_seq
        multichain_mask_2d=multichain_mask) # n_seq x n_seq, dtype=bool
      pair_activations += template_act
    if self.timmers:
      self.timmers.end_timmer('multimer_embedding')
      self.timmers.save()

    # Extra MSA stack.
    if self.timmers:
      self.timmers.add_timmer('extra_msa')
    (extra_msa_feat, # (n_msa - 508) x n_seq x 25
     extra_msa_mask) = create_extra_msa_feature(
      extra_msa, # (n_msa - 508) x n_seq
      extra_msa_mask, # (n_msa - 508) x n_seq
      extra_deletion_matrix, # (n_msa - 508) x n_seq
      c['num_extra_msa']) # 2048
    extra_msa_activations = self.extra_msa_activations(
      extra_msa_feat).to(dtype=dtype)
    extra_msa_mask = extra_msa_mask.to(dtype=dtype)
    for extra_msa_iter in self.extra_msa_stack:
      extra_msa_output = extra_msa_iter(
        extra_msa_activations, # 
        pair_activations, # n_seq x n_seq x 128
        extra_msa_mask,
        mask_2d)
      ### update input for next step
      extra_msa_activations = extra_msa_output['msa']
      pair_activations = extra_msa_output['pair']
    if self.timmers:
      self.timmers.end_timmer('extra_msa')
      self.timmers.save()

    # Get the size of the MSA before potentially adding templates, so we
    # can crop out the templates later.
    if self.timmers:
      self.timmers.add_timmer('evoformer_iteration')
    num_msa_sequences = msa_activations.shape[0]
    evoformer_input = {
      'msa': msa_activations,
      'pair': pair_activations,
    }
    evoformer_masks = {
      'msa': msa_mask.to(dtype=dtype),
      'pair': mask_2d
    }
    if c['template']['enabled']:
      template_features, template_masks = self.template_embedding_1d(
        template_aatype,
        template_all_atom_positions,
        template_all_atom_mask
      )
      evoformer_input['msa'] = torch.concat(
        [evoformer_input['msa'], template_features], dim=0)
      evoformer_masks['msa'] = torch.concat(
        [evoformer_masks['msa'], template_masks], dim=0)
      
    # Evoformer iterations
    for evoformer_iter in self.evoformer_iteration:
      evoformer_input = evoformer_iter(
        evoformer_input['msa'],
        evoformer_input['pair'],
        evoformer_masks['msa'],
        evoformer_masks['pair']
      )
    evoformer_output = evoformer_input
    msa_activations = evoformer_output['msa']
    pair_activations = evoformer_output['pair']
    single_activations = self.single_activations(msa_activations[0])
    if self.timmers:
      self.timmers.end_timmer('evoformer_iteration')
      self.timmers.save()

    output = {
        'single': single_activations,
        'pair': pair_activations,
        'msa': msa_activations[:num_msa_sequences, :, :],
        'msa_first_row': msa_activations[0]}
    # Convert back to float32 if we're not saving memory.
    if not gc['bfloat16_output']:
      for k, v in output.items():
        if v.dtype == torch.bfloat16:
          output[k] = v.to(dtype=torch.float32)
    return output


class AlphaFoldIteration(nn.Module):
  """A single recycling iteration of AlphaFold architecture.

  Computes ensembled (averaged) representations from the provided features.
  These representations are then passed to the various heads
  that have been requested by the configuration file.
  """

  def __init__(self, 
    config, 
    global_config, 
    struct_apply):
    super().__init__()
    self.c = config
    self.gc = global_config
    self.embedding_module = EmbeddingsAndEvoformer(
      self.c['embeddings_and_evoformer'], 
      self.gc)
    self.heads = OrderedDict()
    for head_name, head_config in sorted(self.c['heads'].items()):
      if head_name in ['structure_module']:
        continue  # Do not instantiate zero-weight heads.
      head_factory = {
          'masked_msa': MaskedMsaHead,
          'distogram': DistogramHead,
          'predicted_lddt': PredictedLDDTHead,
          'predicted_aligned_error': PredictedAlignedErrorHead,
          'experimentally_resolved': ExperimentallyResolvedHead,
      }[head_name]
      self.heads[head_name] = (head_factory(head_config, self.gc))
    self.heads['structure_module'] = struct_apply

  def forward(self,
      batch,
      Struct_Params,
      rng):
    num_ensemble = torch.tensor(self.c['num_ensemble_eval'])
    # Compute representations for each MSA sample and average.
    representations = {}
    for i_ensemble in range(num_ensemble): # =1
      if i_ensemble == 0:
        representations = self.embedding_module(**batch)
      else:
        representations_update = self.embedding_module(**batch)
        for k, v in representations_update.items():
          if k not in {'msa', 'true_msa', 'bert_mask'}:
            representations[k] += v * (1./num_ensemble).to(dtype=v.dtype)
          else: # non-deterministic point
            representations[k] = v
    self.representations = representations
    self.batch = batch
    ret = OrderedDict(representations = representations)
    # StructureModule head
    if Struct_Params is not None:
      representations_hk = jax.tree_map(detached, representations)
      batch_hk = jax.tree_map(detached, batch)
      res_hk = self.heads['structure_module'](
        Struct_Params, rng, representations_hk, batch_hk)
      ret['structure_module'] = jax.tree_map(list2tensor, res_hk)
      if 'act' in ret['structure_module'].keys():
        representations['structure_module'] = ret['structure_module'].pop('act')
        print('# ====> [INFO] pLDDTHead input has been saved.')
        f_tmp_plddt = 'structure_module_input.pkl'
        while os.path.isfile(f_tmp_plddt):
          f_tmp_plddt = f_tmp_plddt + '-1.pkl'
        with open(f_tmp_plddt, 'wb') as h_tmp:
          pickle.dump(representations['structure_module'], h_tmp, protocol=4)
    # masked_msa & distogram
    for name in ['masked_msa', 'distogram']:
      ret[name] = self.heads[name](representations)

    # Add confidence heads after StructureModule is executed.
    ret['predicted_lddt'] = self.heads['predicted_lddt'](representations)
    ret['experimentally_resolved'] = self.heads['experimentally_resolved'](representations)
    ret['predicted_aligned_error'] = self.heads['predicted_aligned_error'](representations)
      # Will be used for ipTM computation.
    ret['predicted_aligned_error']['asym_id'] = batch['asym_id']
    return ret


class AlphaFold(object):
  def __init__(self, 
    config, 
    af2iter_params,
    struct_apply,
    struct_params,
    struct_rng,
    name='alphafold') -> None:
    super().__init__()
    self.c = config
    self.gc = config['global_config']
    self.impl = AlphaFoldIteration(self.c, self.gc, struct_apply)
    if af2iter_params is not None:
      af2iter_params = filtered_pth_params(af2iter_params, self.impl)
      self.impl.load_state_dict(af2iter_params)
    self.impl.eval()
    self.struct_params = struct_params
    self.struct_rng = struct_rng

  def _get_prev(self, ret):
    new_prev = {}
    if 'structure_module' in ret.keys():
      new_prev['prev_pos'] = ret['structure_module']['final_atom_positions']
    else:
      num_residues = ret['representations']['msa'].shape[1]
      new_prev['prev_pos'] = torch.zeros((num_residues, 37, 3))
    new_prev['prev_msa_first_row'] = ret['representations']['msa_first_row']
    new_prev['prev_pair'] = ret['representations']['pair']
    return new_prev

  def _distances(self, points):
    return torch.sqrt(torch.sum(
      (points[:, None] - points[None, :])**2, dim=-1))

  def _recycle_cond(self, 
    idx_iter, 
    prev,
    next_in,
    batch, 
    num_iter):
    ca_idx = atom_order['CA']
    sq_diff = torch.square(
      self._distances(prev['prev_pos'][:, ca_idx, :]) - 
      self._distances(next_in['prev_pos'][:, ca_idx, :]))
    mask = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]
    sq_diff = mask_mean_simple(mask, sq_diff, axis=None)
    # Early stopping criteria based on criteria used in
    # AF2Complex: https://www.nature.com/articles/s41467-022-29394-2
    diff = torch.sqrt(sq_diff + 1e-8)
    less_than_max_recycles = (idx_iter < num_iter)
    has_exceeded_tolerance = (
      (idx_iter == 0) | (diff > self.c['recycle_early_stop_tolerance']))
    return less_than_max_recycles & has_exceeded_tolerance

  def __call__(self, 
    batch,
    return_representations=False):
    c = self.c
    num_res = batch['aatype'].shape[0] # [Nseq]
    prev = {}
    emb_config = self.c['embeddings_and_evoformer']
    if emb_config['recycle_pos']:
      prev['prev_pos'] = torch.zeros((num_res, atom_type_num, 3)) # Nres x 37 x 3
    if emb_config['recycle_features']:
      prev['prev_msa_first_row'] = torch.zeros(num_res, emb_config['msa_channel']) # Nres x 256
      prev['prev_pair'] = torch.zeros((num_res, num_res, emb_config['pair_channel'])) # Nres x Nres x 128
    if self.c['num_recycle']:
      num_recycles = self.c['num_recycle']
      if 'num_iter_recycling' in batch:
        num_iter = batch['num_iter_recycling'][0]
        num_iter = torch.minimum(num_iter, c['num_recycle'])
      else:
        num_iter = c['num_recycle'] # 3
      for idx_iter in range(0, num_iter+1):
        for k, v in prev.items():
          batch[k] = v
        with torch.inference_mode():
          res = self.impl(batch, self.struct_params, self.struct_rng)
        if idx_iter < num_iter:
          next_in = self._get_prev(res)
        idx_iter_1 = idx_iter + 1
        print(f'# [INFO] recycl: {idx_iter_1}/{num_iter}')
        if not self._recycle_cond(
          idx_iter, prev, next_in, batch, num_recycles):
          print('# [INFO] Result is stable so that we can successfully stop earlier.')
          break
        prev = next_in
    else:
      num_recycles = 0
      for k, v in prev.items():
        batch[k] = v
      with torch.inference_mode():
        res = self.impl(batch, self.struct_params, self.struct_rng)
    if not return_representations:
      del res['representations']
    res['num_recycles'] = num_recycles
    return res

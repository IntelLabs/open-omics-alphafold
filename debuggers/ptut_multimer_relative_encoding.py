import torch
from alphafold.model.config import model_config
from torch.nn import functional as F
import pdb


mc = model_config('model_1_multimer_v3')['model']
n_msa = 1230
max_seq = mc['embeddings_and_evoformer']['num_msa'] # 508
n_seq = 254
residue_index = torch.ones((n_seq), dtype=torch.int64)
asym_id = torch.ones((n_seq), dtype=torch.int64)
entity_id = torch.ones((n_seq), dtype=torch.int64)
sym_id = torch.ones((n_seq), dtype=torch.int64)


def _relative_encoding( 
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
  c = mc['embeddings_and_evoformer']
  gc = mc['global_config']
  rel_feats = []
  pos = residue_index
  asym_id_same = asym_id[:, None] == asym_id[None, :]
  offset = pos[:, None] - pos[None, :] # n_seq x n_seq
  dtype = torch.bfloat16 if gc['bfloat16'] else torch.float32
  clipped_offset = torch.clip(
    offset + c['max_relative_idx'], 0, 2 * c['max_relative_idx']) # 2 * 32

  if c['use_chain_relative']: # True
    final_offset = torch.where(asym_id_same, clipped_offset,
                             (2 * c['max_relative_idx'] + 1) *
                             torch.ones_like(clipped_offset))
    rel_pos = F.one_hot(final_offset, 2 * c['max_relative_idx'] + 2)
    rel_feats.append(rel_pos)
    entity_id_same = entity_id[:, None] == entity_id[None, :]
    rel_feats.append(entity_id_same.to(dtype=rel_pos.dtype)[..., None])
    rel_sym_id = sym_id[:, None] - sym_id[None, :]
    max_rel_chain = c['max_relative_chain'] # 2
    clipped_rel_chain = torch.clip(
        rel_sym_id + max_rel_chain, 0, 2 * max_rel_chain)
    final_rel_chain = torch.where(entity_id_same, clipped_rel_chain,
                                (2 * max_rel_chain + 1) *
                                torch.ones_like(clipped_rel_chain))
    rel_chain = F.one_hot(final_rel_chain, 2 * c['max_relative_chain'] + 2)
    rel_feats.append(rel_chain)
  else:
    rel_pos = F.one_hot(clipped_offset, 2 * c['max_relative_idx'] + 1)
    rel_feats.append(rel_pos)
  rel_feat = torch.concat(rel_feats, dim=-1)
  rel_feat = rel_feat.to(dtype)
  return rel_feat # n_seq x n_seq x input_feat


rel_feat = _relative_encoding(
  residue_index,
  asym_id,
  entity_id,
  sym_id)
print('rel_feat =', rel_feat.shape)


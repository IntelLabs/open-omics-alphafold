from alphafold_pytorch_jit.basics import GatingAttention
from pcl_pytorch_extension.alphafold.Alpha_Attention import GatingAttentionOpti_forward
GatingAttention.forward = GatingAttentionOpti_forward


from alphafold_pytorch_jit.subnets_multimer import AlphaFoldIteration
import torch
from alphafold_pytorch_jit.folding_multimer import StructureModule
from alphafold.model.config import model_config
import pdb
from time import time
import os
from jax.random import PRNGKey
from alphafold_pytorch_jit.weight_io import load_npy2hk_params
from alphafold_pytorch_jit.hk_io import get_pure_fn


mc = model_config('model_1_multimer_v3')['model']
gc = mc['global_config']
n_msa = 1230
n_seq = 254
random_seed = 123
root_weights = '/home/yangw/weights/extracted/model_1_multimer_v3/alphafold/alphafold_iteration'
root_struct = os.path.join(root_weights, 'structure_module')
struct_params = load_npy2hk_params(root_struct)
struct_rng = PRNGKey(random_seed)
_, struct_apply = get_pure_fn(
  StructureModule, mc['heads']['structure_module'], gc)
batch = dict(
  msa = torch.ones((n_msa, n_seq), dtype=torch.int64),
  msa_mask = torch.ones((n_msa, n_seq), dtype=torch.float32),
  seq_mask = torch.ones((n_seq), dtype=torch.float32),
  aatype = torch.ones((n_seq), dtype=torch.int64),
  cluster_bias_mask = torch.ones((n_msa), dtype=torch.float32),
  deletion_matrix = torch.ones((n_msa, n_seq), dtype=torch.float32),
  bert_mask = torch.ones((n_msa, n_seq), dtype=torch.float32),
  template_aatype = torch.ones((4, n_seq), dtype=torch.int64),
  template_all_atom_mask = torch.ones((4, n_seq, 37), dtype=torch.float32),
  template_all_atom_positions = torch.ones((4, n_seq, 37, 3), dtype=torch.float32),
  residue_index = torch.ones((n_seq), dtype=torch.int64),
  asym_id = torch.ones((n_seq), dtype=torch.int64),
  entity_id = torch.ones((n_seq), dtype=torch.int64),
  sym_id = torch.ones((n_seq), dtype=torch.int64),
  prev_pos = torch.ones((n_seq, 37, 3), dtype=torch.float32),
  prev_msa_first_row = torch.ones((n_seq, 256), dtype=torch.float32),
  prev_pair = torch.ones((n_seq, n_seq, 128), dtype=torch.float32)
)

model = AlphaFoldIteration(mc, gc, struct_apply)
model.eval()

t0 = time()
with torch.inference_mode():
  res = model(batch, struct_params, struct_rng)
print(time() - t0)

for k, v in res.items():
  if isinstance(v, dict):
    for k2, v2 in v.items():
      print('{}.{} = {}'.format(k, k2, v2.shape))
  else:
    print('{} = {}'.format(k, v.shape))
pdb.set_trace()


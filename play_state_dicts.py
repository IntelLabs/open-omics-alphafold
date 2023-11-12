from alphafold_pytorch_jit.subnets_multimer import AlphaFoldIteration
from alphafold.model.config import model_config
import numpy as np
import os
from jax.random import PRNGKey
from alphafold_pytorch_jit.weight_io import load_npy2hk_params
from alphafold_pytorch_jit.hk_io import get_pure_fn
from alphafold_pytorch_jit.folding_multimer import StructureModule
import pdb


mc = model_config('model_1_multimer_v3')['model']
gc = mc['global_config']
random_seed = 123
root_weights = '/home/yangw/weights/extracted/model_1_multimer_v3/alphafold/alphafold_iteration'
root_struct = os.path.join(root_weights, 'structure_module')
struct_params = load_npy2hk_params(root_struct)
struct_rng = PRNGKey(random_seed)
_, struct_apply = get_pure_fn(
  StructureModule, mc['heads']['structure_module'], gc)
model = AlphaFoldIteration(mc, gc, struct_apply)
model.eval()
states = model.state_dict()
i = 0
for k in states.keys():
  if 'extra_msa' in k and i < 1000:
    print(k)
    i+=1

f_weight = '/mnt/data1/params_2022/params_model_1_multimer_v3.npz'
df_weights = np.load(f_weight)
i=0
for k in df_weights.keys():
  if 'extra_msa' in k and i < 1000:
    print(k)
    i +=1

pdb.set_trace()

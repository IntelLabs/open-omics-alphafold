import os
from alphafold_pytorch_jit.weight_io import load_npy2pth_params, load_npy2hk_params, filtered_pth_params, fix_multimer_params
from alphafold_pytorch_jit.folding import StructureModule
from alphafold_pytorch_jit.hk_io import get_pure_fn
from alphafold_pytorch_jit import subnets_multimer
import jax
from alphafold.model.config import model_config


model_name='model_1_multimer_v3'
root_params = f'/mnt/remote6/yangw/af2home/weights/extracted/{model_name}'
root_af2iter = os.path.join(root_params, 'alphafold/alphafold_iteration') # validated
root_struct = os.path.join(root_af2iter, 'structure_module')

af2iter_params = load_npy2pth_params(root_af2iter)
struct_params = load_npy2hk_params(root_struct)
cfg = model_config(model_name)
mc = cfg['model']
gc = mc['global_config']
sc = mc['heads']['structure_module']
is_multimer = gc['multimer_mode']
print('#### is_multimer =', is_multimer)

struct_rng = jax.random.PRNGKey(123)
_, struct_apply = get_pure_fn(StructureModule, sc, gc)


impl = subnets_multimer.AlphaFoldIteration(mc, gc, struct_apply)
fix_multimer_params(af2iter_params, impl)

# af2iter_params = filtered_pth_params(af2iter_params, impl)
# impl.load_state_dict(af2iter_params)

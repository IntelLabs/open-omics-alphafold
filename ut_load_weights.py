# for RunModel
from alphafold_pytorch_jit.net import RunModel


# for AlphaFold
from alphafold_pytorch_jit import subnets_multimer
from alphafold.model.config import model_config

# for AlphaFoldIteration
import os
from alphafold_pytorch_jit.weight_io import load_params
from alphafold_pytorch_jit.folding import StructureModule
from alphafold_pytorch_jit.hk_io import get_pure_fn
import jax

model_name='model_1_multimer_v3'
root_params = f'/mnt/remote6/yangw/af2home/weights/extracted/{model_name}'

cfg = model_config(model_name)
mc = cfg['model']

# test RunModel
RunModel(mc, root_params, )

# test alphafold
model = subnets_multimer.AlphaFold(mc)
model.load_weights(root_params)

# test alphafolditeration
# gc = mc['global_config']
# is_multimer = gc['multimer_mode']
# print('#### is_multimer =', is_multimer)
# sc = mc['heads']['structure_module']
# struct_rng = jax.random.PRNGKey(123)
# _, struct_apply = get_pure_fn(StructureModule, sc, gc)
# af2iter_params, head_params = load_params(root_params, 'multimer')
# impl = subnets_multimer.AlphaFoldIteration(mc, gc, struct_apply)
# impl.load_backbone_params(af2iter_params)
# impl.load_head_params(head_params)

import pickle as pkl
import pdb
import sys
import torch
sys.path.append('/home/yangw/sources/intel-alphafold2')
import os
from alphafold_pytorch_jit.heads import PredictedLDDTHead as pth_head
from alphafold_haiku.model.modules import PredictedLDDTHead as hk_head
from alphafold_haiku.model.config import model_config
from alphafold_haiku.common import confidence
from alphafold_pytorch_jit.weight_io import load_npy2pth_params, load_npy2hk_params
from alphafold_pytorch.tools.hk_io import get_pure_fn
import jax
from numpy.testing import assert_almost_equal

# load params
module_prefix = 'predicted_lddt'
root_weights = '/home/yangw/weights/extracted/model_1/alphafold/alphafold_iteration/predicted_lddt_head'
root_module = '/home/yangw/sources/intel-alphafold2/debuggers'
f_log = os.path.join(root_module, 'test_%s.txt' % module_prefix)
f_trace = os.path.join(root_module, 'test_%s.json' % module_prefix)
cfg = model_config('model_1')
c = cfg['model']['heads'][module_prefix]
gc = cfg['model']['global_config']

# load sample
f_sample = 'structure_module_input.pkl'
with open(f_sample, 'rb') as h:
  sample = pkl.load(h)
  sample_pth = {'structure_module':sample}
sample_hk = {
  'structure_module':sample.detach().cpu().numpy()}

# build PTH model
pth_model = pth_head(c, gc)
params_dict = load_npy2pth_params(root_weights)
pth_model.load_state_dict(params_dict)
pth_model.eval()

# build HK model
init, apply = get_pure_fn(hk_head, c, gc)
params_dict = load_npy2hk_params(root_weights)

# infer by PTH
with torch.no_grad():
  res_pth = pth_model(sample_pth)
plddts_pth = confidence.compute_plddt(res_pth['logits'].detach().cpu().numpy())

# infer by HK
rng = jax.random.PRNGKey(0)
init(rng, sample_hk, None, False)
res_hk = apply(params_dict, rng, sample_hk, None, False)
plddts_hk = confidence.compute_plddt(res_hk['logits'])

# compare results of PTH v.s. HK
is_equal = assert_almost_equal(plddts_pth, plddts_hk, decimal=3)
diff = abs((plddts_hk - plddts_pth).sum()) / len(plddts_pth)
print('# [INFO] is_equal(plddts_pytorch, plddts_hk) =',is_equal)
print('# average difference = %.6f' % diff)

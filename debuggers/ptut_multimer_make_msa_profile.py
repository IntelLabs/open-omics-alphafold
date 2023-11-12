from alphafold_pytorch_jit.basics_multimer import make_msa_profile
import torch
from alphafold.model.config import model_config
import pdb


mc = model_config('model_1_multimer_v3')['model']
n_msa = 1230
n_seq = 254
msa = torch.ones((n_msa, n_seq), dtype=torch.int64)
msa_mask = torch.ones((n_msa, n_seq), dtype=torch.float32)

msa_profile = make_msa_profile(msa,msa_mask)
print('msa_profile =', msa_profile.shape)
pdb.set_trace()


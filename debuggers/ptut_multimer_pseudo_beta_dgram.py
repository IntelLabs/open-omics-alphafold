from alphafold_pytorch_jit.basics_multimer import pseudo_beta_fn
from alphafold_pytorch_jit.basics import dgram_from_positions_pth
import torch
from alphafold.model.config import model_config
import pdb


mc = model_config('model_1_multimer_v3')['model']
n_msa = 1230
n_seq = 254
aatype = torch.ones((n_seq), dtype=torch.int64)
prev_pos = torch.ones((n_seq, 37, 3), dtype=torch.float32)
all_atom_masks = None
dtype = torch.float16


prev_pseudo_beta = pseudo_beta_fn(aatype,prev_pos, all_atom_masks)
dgram = dgram_from_positions_pth(prev_pseudo_beta, **mc['embeddings_and_evoformer']['prev_pos'])
dgram = dgram.to(dtype=dtype)
print('prev_pseudo_beta =', prev_pseudo_beta.shape)
print('dgram =', dgram.shape)
pdb.set_trace()


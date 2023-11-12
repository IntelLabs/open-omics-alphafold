from alphafold_pytorch_jit.basics_multimer import make_masked_msa
import torch
from alphafold.model.config import model_config
import pdb


mc = model_config('model_1_multimer_v3')['model']
cfg_maked_msa = mc['embeddings_and_evoformer']['masked_msa']
n_msa = 1230
n_seq = 254
msa = torch.ones((n_msa, n_seq), dtype=torch.int64)
msa_mask = torch.ones((n_msa, n_seq), dtype=torch.int64)
msa_profile = torch.ones((n_seq, 22), dtype=torch.float32)
bert_mask = torch.ones((n_msa, n_seq), dtype=torch.float32)

msa, bert_mask, true_msa = make_masked_msa(
  msa, msa_mask, msa_profile, cfg_maked_msa, bert_mask)
print('true_msa =', true_msa.shape)
pdb.set_trace()


from alphafold_pytorch_jit.basics_multimer import create_extra_msa_feature
import torch
from alphafold.model.config import model_config
import pdb


mc = model_config('model_1_multimer_v3')['model']
n_msa = 1230
max_seq = mc['embeddings_and_evoformer']['num_msa'] # 508
n_seq = 254
extra_msa = torch.ones((n_msa-max_seq, n_seq), dtype=torch.int64)
extra_msa_mask = torch.ones((n_msa-max_seq, n_seq), dtype=torch.float32)
extra_deletion_matrix = torch.ones((n_msa-max_seq, n_seq), dtype=torch.float32)


extra_msa_feat, extra_msa_mask = create_extra_msa_feature(
  extra_msa,
  extra_msa_mask,
  extra_deletion_matrix,
  mc['embeddings_and_evoformer']['num_extra_msa'])
print('extra_msa_feat =', extra_msa_feat.shape)
print('extra_msa_mask =', extra_msa_mask.shape)
print(mc['embeddings_and_evoformer']['extra_msa_channel'])
pdb.set_trace()


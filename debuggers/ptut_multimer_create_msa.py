from alphafold_pytorch_jit.basics_multimer import create_msa_feat
import torch
from alphafold.model.config import model_config
import pdb


mc = model_config('model_1_multimer_v3')['model']
n_msa = 1230
n_seq = 254
msa = torch.ones((n_msa, n_seq), dtype=torch.int64)
deletion_matrix = torch.ones((n_msa, n_seq), dtype=torch.float32)
cluster_deletion_mean = torch.ones((n_msa, n_seq), dtype=torch.float32)
cluster_profile = torch.ones((n_msa, n_seq, 23), dtype=torch.float32)

preprocess_1d = torch.ones((n_seq, 256), dtype=torch.float32)
preprocess_msa = torch.ones((n_msa, n_seq, 256))


msa_feat = create_msa_feat(
  msa,deletion_matrix, cluster_deletion_mean, cluster_profile)
msa_activations = torch.unsqueeze(preprocess_1d, dim=0) + preprocess_msa
print('msa_feat =', msa_feat.shape)
print('msa_activations =', msa_activations.shape)
pdb.set_trace()


from alphafold_pytorch_jit.basics_multimer import nearest_neighbor_clusters
import torch
from alphafold.model.config import model_config
import pdb


mc = model_config('model_1_multimer_v3')['model']
n_msa = 1230
n_seq = 254
msa = torch.ones((n_msa, n_seq), dtype=torch.int64)
msa_mask = torch.ones((n_msa, n_seq), dtype=torch.float32)
deletion_matrix = torch.ones((n_msa, n_seq), dtype=torch.float32)
max_seq = mc['embeddings_and_evoformer']['num_msa'] # = 508
extra_msa = torch.ones((n_msa-max_seq, n_seq), dtype=torch.int64)
extra_msa_mask = torch.ones((n_msa-max_seq, n_seq), dtype=torch.float32)
extra_deletion_matrix = torch.ones((n_msa-max_seq, n_seq), dtype=torch.float32)


cluster_profile, cluster_deletion_mean = nearest_neighbor_clusters(
  msa,msa_mask,deletion_matrix,extra_msa, extra_msa_mask, extra_deletion_matrix)
print('cluster_profile =', cluster_profile.shape)
print('cluster_deletion_mean =', cluster_deletion_mean.shape)
pdb.set_trace()


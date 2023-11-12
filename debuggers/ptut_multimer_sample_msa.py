from alphafold_pytorch_jit.basics_multimer import sample_msa
import torch
from alphafold.model.config import model_config
import pdb


mc = model_config('model_1_multimer_v3')['model']
n_msa = 1230
n_seq = 254
msa = torch.ones((n_msa, n_seq), dtype=torch.float32)
msa_mask = torch.ones((n_msa, n_seq), dtype=torch.float32)
max_seq = mc['embeddings_and_evoformer']['num_msa'] # = 508
cluster_bias_mask = torch.ones((n_msa), dtype=torch.float32)
deletion_matrix = torch.ones((n_msa, n_seq), dtype=torch.float32)
bert_mask = torch.ones((n_msa, n_seq), dtype=torch.float32)


(msa, 
 msa_mask, 
 cluster_bias_mask, 
 extra_msa, 
 extra_msa_mask, 
 deletion_matrix, 
 bert_mask, 
 extra_deletion_matrix, 
 extra_bert_mask
) = sample_msa(
  msa,msa_mask,max_seq,cluster_bias_mask,deletion_matrix,bert_mask)
print('extra_msa =', extra_msa.shape)
print('extra_msa_mask =', extra_msa_mask.shape)
print('extra_deletion_matrix =', extra_deletion_matrix.shape)
print('extra_bert_mask =', extra_bert_mask.shape)
pdb.set_trace()


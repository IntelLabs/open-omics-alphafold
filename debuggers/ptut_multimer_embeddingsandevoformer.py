from alphafold_pytorch_jit.basics import GatingAttention
from pcl_pytorch_extension.alphafold.Alpha_Attention import GatingAttentionOpti_forward
GatingAttention.forward = GatingAttentionOpti_forward


from alphafold_pytorch_jit.subnets_multimer import EmbeddingsAndEvoformer
import torch
from alphafold.model.config import model_config
import pdb
from time import time


mc = model_config('model_1_multimer_v3')['model']
gc = mc['global_config']
n_msa = 1230
n_seq = 254

msa = torch.ones((n_msa, n_seq), dtype=torch.int64)
msa_mask = torch.ones((n_msa, n_seq), dtype=torch.float32)
seq_mask = torch.ones((n_seq), dtype=torch.float32)
aatype = torch.ones((n_seq), dtype=torch.int64)
cluster_bias_mask = torch.ones((n_msa), dtype=torch.float32)
deletion_matrix = torch.ones((n_msa, n_seq), dtype=torch.float32)
bert_mask = torch.ones((n_msa, n_seq), dtype=torch.float32)
template_aatype = torch.ones((4, n_seq), dtype=torch.int64)
template_all_atom_mask = torch.ones((4, n_seq, 37), dtype=torch.float32)
template_all_atom_positions = torch.ones((4, n_seq, 37, 3), dtype=torch.float32)
residue_index = torch.ones((n_seq), dtype=torch.int64)
asym_id = torch.ones((n_seq), dtype=torch.int64)
entity_id = torch.ones((n_seq), dtype=torch.int64)
sym_id = torch.ones((n_seq), dtype=torch.int64)
prev_pos = torch.ones((n_seq, 37, 3), dtype=torch.float32)
prev_msa_first_row = torch.ones((n_seq, 256), dtype=torch.float32)
prev_pair = torch.ones((n_seq, n_seq, 128), dtype=torch.float32)


model = EmbeddingsAndEvoformer(mc['embeddings_and_evoformer'], gc)
model.eval()


t0 = time()
with torch.inference_mode():
  res = model(
    msa,
    msa_mask,
    seq_mask,
    aatype,
    cluster_bias_mask,
    deletion_matrix,
    bert_mask,
    template_aatype,
    template_all_atom_mask, 
    template_all_atom_positions,
    residue_index,
    asym_id,
    entity_id,
    sym_id,
    prev_pos,
    prev_msa_first_row,
    prev_pair
  )
print(time() - t0)

for k, v in res.items():
  print('{} = {}'.format(k, v.shape))
pdb.set_trace()


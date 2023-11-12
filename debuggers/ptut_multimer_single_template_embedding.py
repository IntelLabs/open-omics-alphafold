from alphafold_pytorch_jit.embeddings_multimer import SingleTemplateEmbedding
from alphafold_haiku.model.config import model_config
import torch
from tqdm import tqdm

num_channel = 128
a_dim = 128
num_res = 206
query_embedding = torch.ones([num_res, num_res, num_channel], dtype=torch.float32)
template_aatype = torch.ones([num_res], dtype=torch.float32)
template_all_atom_positions = torch.ones([num_res, 37, 3], dtype=torch.float32)
template_all_atom_mask = torch.ones([num_res, 37], dtype=torch.float32)
padding_mask_2d = torch.ones([num_res, num_res], dtype=torch.float32)
multichain_mask_2d = torch.ones([num_res, num_res], dtype=torch.float32)
cfg = model_config('model_1_multimer_v3')
c = cfg['model']['embeddings_and_evoformer']['template']
gc = cfg['model']['global_config']

model = SingleTemplateEmbedding(c,gc)
model.eval()

with torch.inference_mode():
  for i in tqdm(range(100), desc='infer of SingleTemplateEmbedding'):
    act = model(
      query_embedding,
      template_aatype,
      template_all_atom_positions,
      template_all_atom_mask,
      padding_mask_2d,
      multichain_mask_2d)

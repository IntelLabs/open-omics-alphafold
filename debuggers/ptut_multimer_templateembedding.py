from alphafold_pytorch_jit.embeddings_multimer import TemplateEmbedding
from alphafold_haiku.model.config import model_config
import torch
from tqdm import tqdm
import intel_extension_for_pytorch as ipex
import pdb
from time import time


num_res = 764
query_embedding = torch.ones([num_res, num_res, 128], dtype=torch.float32)
template_aatype = torch.ones([4, num_res], dtype=torch.float32)
template_all_atom_positions = torch.ones([4, num_res, 37, 3], dtype=torch.float32)
template_all_atom_mask = torch.ones([4, num_res, 37], dtype=torch.float32)
padding_mask_2d = torch.ones([num_res, num_res], dtype=torch.float32)
multichain_mask_2d = torch.ones([num_res, num_res], dtype=torch.bool)
cfg = model_config('model_1_multimer_v3')
num_pair_channel = cfg['model']['embeddings_and_evoformer']['pair_channel']
print(cfg['model']['embeddings_and_evoformer'].keys())
c = cfg['model']['embeddings_and_evoformer']['template']
gc = cfg['model']['global_config']

model = TemplateEmbedding(c, gc, num_pair_channel)
model.eval()


with torch.inference_mode(), torch.no_grad():
  for i in tqdm(range(3), desc='infer of TemplateEmbedding'):
    t0 = time()
    template_act = model(
      query_embedding,
      template_aatype,
      template_all_atom_positions,
      template_all_atom_mask,
      padding_mask_2d,
      multichain_mask_2d)
    if i > 1:
      dt = time() - t0
      print(f'duration = {dt} sec')
print(template_act.shape)
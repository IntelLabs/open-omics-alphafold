from alphafold_pytorch_jit.embeddings_multimer import TemplateEmbeddingIteration
from alphafold_haiku.model.config import model_config
import torch
from tqdm import tqdm

num_channel = 64
a_dim = 64
num_res = 764
sample_act = torch.ones([num_res, num_res, num_channel], dtype=torch.float32)
sample_mask = torch.ones([num_res, num_res], dtype=torch.float32)
cfg = model_config('model_1')
c = cfg['model']['embeddings_and_evoformer']['template']['template_pair_stack']
gc = cfg['model']['global_config']

model = TemplateEmbeddingIteration(c,gc)
model.eval()

with torch.inference_mode():
  for i in tqdm(range(10), desc='infer of TemplateEmbeddingIteration'):
    _ = model(sample_act, sample_mask)

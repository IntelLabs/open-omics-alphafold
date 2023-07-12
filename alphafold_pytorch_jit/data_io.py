import torch
import numpy as np
import jax
import pickle

def data_jax2pth(f):
  assert f.endswith('.pkl')
  with open(f, 'rb') as h:
    d = pickle.load(h)
  return jax.tree_map(lambda x: torch.tensor(np.array(x)), d)


def eval_shape(data:dict):
  return jax.tree_map(lambda x: x.shape, data)


if __name__ == '__main__':
  f = '/lustre/home/acct-stu/stu222/experiments/cmp/embeddings/jax_result.pkl'
  df = data_jax2pth(f)
  import pdb
  pdb.set_trace()
from typing import Mapping
import numpy as np
import os

FeatureDict = Mapping[str, np.ndarray]

def save_feature_dict(f:str, data:FeatureDict):
  np.savez(f, **data)

def load_feature_dict(f:str) -> FeatureDict:
  df = np.load(f, allow_pickle=True)
  res = {}
  for k in df.files:
    res[k] = df[k]
  return res

def load_feature_dict_if_exist(f:str):
  if os.path.exists(f):
    return load_feature_dict(f)
  else:
    return None

def get_mock_2darray(h, w):
  np.random.seed(1)
  if h == 0:
    return np.array(['type1', 'type2', 'type3'])
  else:
    return np.random.random((h, w)) if h % 4 == 0 else (np.random.random((h, w)) * 255).astype(np.int)

if __name__ == '__main__':
  f = r'C:\Users\wyang2\datasets\af2_sample\features.pkl'
  data = None
  import pickle
  with open(f, 'rb') as h:
    data = pickle.load(h)
  
  import pdb
  pdb.set_trace()
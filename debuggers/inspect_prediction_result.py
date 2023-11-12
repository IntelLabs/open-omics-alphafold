import pickle as pkl
import pdb
import sys
sys.path.append('/home/yangw/sources/intel-alphafold2')



#f = '/mnt/data1/demohome/experiments/debug/spike/result_model_1.pkl'
f = '/home/yangw/experiments/mmcif_6yke/result_model_1.pkl'

with open(f, 'rb') as h:
  df = pkl.load(h)
  pdb.set_trace()
  plddt = df['plddt']
print(plddt)
print(df.keys())
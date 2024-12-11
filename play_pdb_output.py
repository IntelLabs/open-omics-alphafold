import pickle
from alphafold.common import protein
import numpy as np
import os
from runners.saver import load_feature_dict_if_exist
from alphafold.common.residue_constants import atom_type_num


root_result = '/data/yangw/af2home/experiments_multimer/igmfc'
fp_result = os.path.join(root_result, 'result_model_1_multimer_v3_pred_0.pkl')
fp_features = os.path.join(root_result, 'intermediates/features.npz')
model_name = 'model_1_multimer_v3_pred_0'
unrelaxed_pdb_path = os.path.join(root_result, f'unrelaxed_{model_name}_rank0.pdb')

with open(fp_result, 'rb') as h:
  prediction_result = pickle.load(h)
  plddt = prediction_result['plddt']
  plddt_b_factors = np.repeat(plddt[:, None], atom_type_num, axis=-1)
df_features = load_feature_dict_if_exist(fp_features)

unrelaxed_proteins = {}
unrelaxed_pdbs = {}
unrelaxed_protein = protein.from_prediction(
  df_features, 
  prediction_result,
  plddt_b_factors,
  remove_leading_feature_dimension=False)
unrelaxed_proteins[model_name] = unrelaxed_protein
unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
with open(unrelaxed_pdb_path, 'w') as h:
  h.write(unrelaxed_pdbs[model_name]) # save unrelaxed pdb

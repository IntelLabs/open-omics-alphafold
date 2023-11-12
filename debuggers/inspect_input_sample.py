import sys
sys.path.append('/home/yangw/sources/af2pth')
from runners.saver import load_feature_dict_if_exist
import os
import pdb


root_path = '/home/yangw/experiments/debug/mmcif_3geh_0/intermediates'
f = os.path.join(root_path, 'processed_features.npz')
df = load_feature_dict_if_exist(f)
evo_init_dims = {
      'target_feat':df['target_feat'].shape[-1], # 固定就是 22，不会随着输入序列长短而变化
      'msa_feat':df['msa_feat'].shape[-1] # 固定就是49，不会随着输入序列长短而变化
}
print(evo_init_dims)
pdb.set_trace()
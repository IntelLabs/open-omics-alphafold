import os
import torch
import numpy as np
import pdb
from collections import OrderedDict


hk2pth_name  = {'scale':'weight', 'weights':'weight', 'offset': 'bias'}
iters_name = {'extra_msa_stack','evoformer_iteration','template_pair_sub_stack'}
shape_changed_name = {'weights'}

def load_npy2pth_params(root_path):
  delim = '/'
  assert os.path.isdir(root_path)
  res = {}
  for parent, _, files in os.walk(root_path):
    for f in files:
      if f.endswith('.npy'):
        fp = os.path.join(parent, f)
        data=np.load(fp)
        f_name = f.rstrip('.npy')
        # switch shape
        if f_name in shape_changed_name and len(data.shape) > 1:
          data = data.swapaxes(-1,-2)
        if f_name in hk2pth_name:
          f_name = hk2pth_name[f_name]
        subparent = parent[len(root_path)+1:]
        jp_flag = 0
        for sub_name in subparent.split(delim):
          # find iterations & change name
          # Before: Iters.Subparent.bias shape(n,x1,x2)
          # After: Iters.1.Subparent.bias shape(x1,x2)
          #        ......
          #        Iters.n.Subparent.bias shape(x1,x2)          
          if sub_name in iters_name:
            jp_flag = 1
            iter_data = torch.from_numpy(data)
            for i in range(data.shape[0]):
              iter_name = subparent.replace(sub_name,sub_name+delim+'{}'.format(i))
              k = os.path.join(iter_name,f_name).replace(delim,'.')
              res[k] = iter_data[i]
        if jp_flag:
          continue
        k = os.path.join(subparent,f_name).replace(delim,'.')
        # if subparent.split('/')
        res[k] = torch.from_numpy(data)
  return res

def load_npy2hk_params(root_path,sample_iter=False):
  """
  get weights from folder npy format
  args: 
      sample_iter: True->return 1 layers params
                   False-> return all layers params
  """
  def sample_leaf_dict_from_stack(root_path):
    """
    return 1 from 1 layer params
    """
    res = {}
    for parent, dirs, files in os.walk(root_path):
      res_sub = {}
      mod_name = os.path.basename(root_path)
      subparent = parent[len(root_path)-len(mod_name):]
      # judge iterable
      iter_flag = 0
      for sub_name in subparent.split('/'):
        if sub_name in iters_name and sub_name != 'template_pair_sub_stack':
          iter_flag = 1
      for f in files:
        if f.endswith('.npy'):
          f_key = f.rstrip('.npy')
          f_path = os.path.join(parent, f)
          if iter_flag == 1:
            # if iterable, for test simple. one layer only
            res_sub[f_key] = np.load(f_path)[0][None]
          else:
            res_sub[f_key] = np.load(f_path)
      res[subparent] = res_sub
    return res

  def get_leaf_dict(root_path,res={}):

    res = {}
    for parent, dirs, files in os.walk(root_path):
      res_sub = {}
      mod_name = os.path.basename(root_path)
      subparent = parent[len(root_path)-len(mod_name):]
      for f in files:
        if f.endswith('.npy'):
          f_key = f.rstrip('.npy')
          f_path = os.path.join(parent, f)
          res_sub[f_key] = np.load(f_path)
      res[subparent] = res_sub
    return res
  assert os.path.isdir(root_path)

  if sample_iter == True :
    res = sample_leaf_dict_from_stack(root_path)
  else:
    res = get_leaf_dict(root_path)
  return res

def filtered_pth_params(pth_params, model:torch.nn.Module):
  res = {}
  for name in list(model.state_dict().keys()):
    res[name] = pth_params[name]
  assert res.keys() == model.state_dict().keys()
  return OrderedDict(res)

def fix_multimer_params(pth_params, model:torch.nn.Module):
  res = {}
  dst_keys = list(model.state_dict().keys())
  src_keys = list(pth_params.keys())
  n_replace, n_omit = 0, 0
  n_valid, n_tbd_miss, n_tbd_tofuse = 0, 0, 0
  to_replace_prefix = ['evoformer'] # 4518/4580
  to_omit_prefix = [
    'structure_module', 
    'experimentally_resolved_head', 
    'predicted_aligned_error_head',
    'distogram_head',
    'masked_msa_head',
    'predicted_lddt_head'] # 62/4580
  tbd_miss_prefix = 'embedding_module.template_embedding.single_template_embedding', # 72/4580
  tbd_tofuse_prefix = [
    'embedding_module.extra_msa_stack',
    'embedding_module.evoformer_iteration',
    'triangle_multiplication_incoming',
    'triangle_multiplication_outgoing'
  ]

  # 权重文件中包含，但模型参数列表里没有的:
    # 'embedding_module.template_embedding.single_template_embedding ****'

  src_key_sample = 'triangle_multiplication_outgoing.projection'
  dst_key_sample = 'extra_msa_stack.0.triangle_multiplication_outgoing.output_projection'
  for k in src_keys:
    prefix = k.split('.')[0]
    if prefix == 'evoformer':
      src_k = k.replace('evoformer.', 'embedding_module.')
      if src_k in dst_keys: # validated 3606/4580
        n_valid +=1
      elif src_k.startswith(tbd_miss_prefix): # 模型参数中不包含的: 72/4580
        n_tbd_miss += 1
      elif (   src_k.startswith(tbd_tofuse_prefix[0]) \
            or src_k.startswith(tbd_tofuse_prefix[1]) \
           ) and \
           ( tbd_tofuse_prefix[2] in src_k \
            or tbd_tofuse_prefix[3] in src_k
           ): # 模型参数显示是split，但应该是fused: 416/4580
        n_tbd_tofuse += 1
      else:
        print(src_k)
        # if src_key_sample in src_k:
        #   print('src_key:', src_k)
          # for dk in dst_keys:
          #   if dst_key_sample in dk:
          #     print('dst_key:', dk)

    elif prefix in to_omit_prefix:
      n_omit += 1
    else:
      print(prefix)
  print(f'valid ratio = {n_valid}/{len(src_keys)}')
  print(f'TBD[miss in dst] ratio = {n_tbd_miss}/{len(src_keys)}')
  print(f'TBD[should fuse] ratio = {n_tbd_tofuse}/{len(src_keys)}')
  print(f'to-replace ratio = {n_replace}/{len(src_keys)}')
  print(f'to-omit ratio = {n_omit}/{len(src_keys)}')


def check_loading_ratio(pth_params, model:torch.nn.Module):
  in_src, in_dst = [], []
  dst_keys = list(model.state_dict().keys())
  src_keys = list(pth_params.keys())
  for name in dst_keys:
    if name in src_keys:
      in_src.append(name)
  for name in src_keys:
    if name in dst_keys:
      in_dst.append(name)
  not_in_src = set(src_keys) - set(in_src)
  not_in_dst = set(dst_keys) - set(in_dst)
  print(f'not in src ratio = {len(not_in_src)}/{len(src_keys)}')
  print(f'not in dst ratio = {len(not_in_dst)}/{len(dst_keys)}')

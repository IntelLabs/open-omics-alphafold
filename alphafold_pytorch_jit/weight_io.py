import os
import torch
import numpy as np
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

def fixed_multimer_backbone_params(pth_params):
  src_keys = list(pth_params.keys())
  n_replace, n_omit = 0, 0
  n_valid, n_tbd_miss, n_tbd_tofuse = 0, 0, 0

  # subgraphs need to re-map
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

  # case-by-case loading
  dst_params = {}
  for k in src_keys:
    prefix = k.split('.')[0]
    if prefix == 'evoformer':
      src_k = k.replace('evoformer.', 'embedding_module.')
      # if src_k in dst_keys: # validated 3606/4580 => evoformer 4518/4580 + head 62/4580
      #   dst_params[src_k] = pth_params[k]
      #   n_valid +=1
      if src_k.startswith(tbd_miss_prefix): # template_embedding stacks: 72
        if 'embedding_module.template_embedding.single_template_embedding.template_embedding_iteration' in src_k:
          # 2 stacks of FusedTriangleMultiplication
          n_stack = pth_params[k].shape[0]
          src_k_prefix = 'embedding_module.template_embedding.single_template_embedding.template_embedding_iteration'
          src_k_suffix = src_k[(len(src_k_prefix)+1):]
          dst_sub_prefix = 'embedding_module.template_module.template_embedder.template_stack'
          for idx in range(n_stack):
            dst_k = f'{dst_sub_prefix}.{idx}.{src_k_suffix}'
            dst_params[dst_k] = pth_params[k][idx]
          n_valid +=1
        elif 'embedding_module.template_embedding.single_template_embedding.template_pair_embedding_' in src_k:
          src_sub_prefix = 'embedding_module.template_embedding.single_template_embedding.template_pair_embedding_'
          idx = int(src_k[len(src_sub_prefix):].split('.')[0])
          src_k_suffix = src_k[len(src_sub_prefix):].split('.')[1]
          dst_sub_prefix = 'embedding_module.template_module.template_embedder.template_pair_embedding_stack'
          dst_k = f'{dst_sub_prefix}.{idx}.{src_k_suffix}'
          if src_k_suffix == 'weight' and pth_params[k].dim() == 1:
            dst_params[dst_k] = torch.unsqueeze(pth_params[k],1)
          else:
            dst_params[dst_k] = pth_params[k]
          n_valid +=1
        elif 'embedding_module.template_embedding.single_template_embedding' in src_k:
          dst_k = src_k.replace(
            'embedding_module.template_embedding.single_template_embedding',
            'embedding_module.template_module.template_embedder'
          )
          dst_params[dst_k] = pth_params[k]
          n_valid += 1
        else:
          print('# [warning] not-found-in-model:', src_k)
          n_tbd_miss += 1
      else:
        if '~_relative_encoding' in src_k:
          dst_k = src_k.replace('~_relative_encoding.', '')
          dst_params[dst_k] = pth_params[k]
          n_valid +=1
        elif 'template_single_embedding' in src_k:
          dst_k = src_k.replace(
            'embedding_module.template_single_embedding',
            'embedding_module.template_embedding_1d.template_single_embedding')
          dst_params[dst_k] = pth_params[k]
          n_valid +=1
        elif 'embedding_module.template_projection' in src_k:
          dst_k = src_k.replace(
            'embedding_module.template_projection',
            'embedding_module.template_embedding_1d.template_projection')
          dst_params[dst_k] = pth_params[k]
          n_valid +=1
        elif 'embedding_module.template_embedding.output_linear' in src_k:
          dst_k = src_k.replace(
            'embedding_module.template_embedding.output_linear',
            'embedding_module.template_module.output_linear')
          dst_params[dst_k] = pth_params[k]
          n_valid +=1
        else:
          dst_params[src_k] = pth_params[k]
          n_valid +=1
    elif prefix in to_omit_prefix: # 62/4580 used in heads, not here
      n_omit += 1
    else:
      print('# [warning] unknown key:', prefix)
  # if n_valid > 0:
  #   print(f'valid ratio = {n_valid}/{len(src_keys)}')
  # if n_tbd_miss > 0:
  #   print(f'keys[miss in dst] ratio = {n_tbd_miss}/{len(src_keys)}')
  # if n_tbd_tofuse > 0:
  #   print(f'keys[should fuse] ratio = {n_tbd_tofuse}/{len(src_keys)}')
  # if n_replace > 0:
  #   print(f'to-replace ratio = {n_replace}/{len(src_keys)}')
  # print(f'# [INFO ] {n_omit}/{len(src_keys)} keys used for head, not here.')
  # print(f'# [INFO] {n_valid}/{len(src_keys)} loaded into AF2 backbone')
  return OrderedDict(dst_params)

def fixed_monomer_backbone_params(pth_params:dict):
  res = {}
  for k in list(pth_params.keys()):
    prefix = k.split('.')[0]
    if prefix == 'evoformer':
      res[k] = pth_params[k]
  return res

def filtered_pth_params(pth_params, model:torch.nn.Module):
  res = {}
  for name in list(model.state_dict().keys()):
      res[name] = pth_params[name]
  assert res.keys() == model.state_dict().keys()
  return OrderedDict(res)

def fixed_head_params(pth_params:dict):
  src_keys = list(pth_params.keys())
  head_prefix = [
    #'structure_module', 
    'experimentally_resolved_head', 
    'predicted_aligned_error_head',
    'distogram_head',
    'masked_msa_head',
    'predicted_lddt_head']
  dst_params = {}
  for k in src_keys:
    prefix = k.split('.')[0]
    if not prefix == 'evoformer':
      v = pth_params[k]
      dst_params[k] = v
  return dst_params

def load_params(root_path, mode):
  root_af2iter = os.path.join(root_path, 'alphafold/alphafold_iteration')
  root_struct = os.path.join(root_af2iter, 'structure_module')
  af2_params = load_npy2pth_params(root_af2iter)
  struct_params = load_npy2hk_params(root_struct)
  af2iter_params = {}
  head_params = {}
  if mode == 'multimer':
    af2iter_params = fixed_multimer_backbone_params(af2_params)
  elif mode == 'monomer':
    af2iter_params = fixed_monomer_backbone_params(af2_params)
  head_params = fixed_head_params(af2_params)
  head_params['structure_module'] = struct_params
  return af2iter_params, head_params

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

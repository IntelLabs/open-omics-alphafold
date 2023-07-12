###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Sasikanth Avancha (Intel Corp.)                                     #
###############################################################################

import torch
from tpp_pytorch_extension._C import _gnn_utils as gnn_utils_cpp


def affinitize_cores(nthreads, nworkers):
    gnn_utils_cpp.affinitize_cores(nthreads, nworkers)


def find_nodes(pnd_s_in, pnd_orig, srcnodes, lnodes, ntype):
    inputs = [pnd_s_in, pnd_orig, srcnodes, lnodes]
    orig, batch, part = gnn_utils_cpp.find_nodes(inputs, ntype)
    return orig, batch, part


def db_r2l_map(db_t, sn_orig, sn_batch, sn_part):
    inputs = [db_t, sn_orig, sn_batch, sn_part]
    r, b, l = gnn_utils_cpp.db_r2l_map(inputs)
    return r, b, l


def r2l_map(o2l_map, rbn_orig):
    inputs = [o2l_map, rbn_orig]
    r_lid2, l_lid2 = gnn_utils_cpp.r2l_map(inputs)
    return r_lid2, l_lid2


def find_n_map_nodes(db_t, pnd_solid, pnd_orig, srcnodes, lnodes):
    inputs = [db_t, pnd_solid, pnd_orig, srcnodes, lnodes]
    r, b, l = gnn_utils_cpp.find_n_map_nodes(inputs)
    return r, b, l


def set_cond_index_vals(inp, cval, idx, outp, oval):
    inputs = [inp, idx, outp]
    gnn_utils_cpp.set_cond_index_vals(inputs, cval, oval)


def set_n_store_cline_indices(rptr, cl, hmap, age, nids, cval, oval):
    inputs = [rptr, cl, hmap, age, nids]
    gnn_utils_cpp.set_n_store_cline_indices(inputs, cval, oval)


def inc_cache_fill(cache_fill, nodes):
    gnn_utils_cpp.inc_cache_fill(cache_fill, nodes)


def cache_load(hmap, oid, feats, age=None, level=0, min_life=0, life=0):
    inputs = [hmap, oid, feats]
    if age is not None:
        inputs.append(age)
    oid_idx, gat_data = gnn_utils_cpp.cache_load(inputs, level, min_life, life)

    return oid_idx, gat_data


def cache_store(cache_data):
    (
        hashmap,
        rptr,
        age,
        nodes,
        storage_feats,
        feats,
        sz_feats,
        feats_sz,
        cp,
        cs,
        hval,
        rval,
    ) = cache_data
    inputs = [hashmap, rptr, age, nodes, storage_feats, feats, sz_feats, feats_sz, cp]
    gnn_utils_cpp.cache_store(inputs, cs, hval, rval)


def node_sampling(degs, xnbn, xrbn, hil, thres):
    inputs = [degs, xnbn, xrbn]
    xnbn, xrbn = gnn_utils_cpp.node_sampling(inputs, hil, thres)
    return xnbn, xrbn


def gather_n_store_offset(inp, ind, out, offi, offv):
    inputs = [inp, ind, out]
    gnn_utils_cpp.gather_n_store_offset(inputs, offi, offv)


def gather_features(nfeat, indices):
    N = indices.shape[0]
    align = 32 if N >= 32 or N == 0 else N
    inputs = [nfeat, indices]

    out = gnn_utils_cpp.gather_features(align, inputs)
    return out


def scatter_features(feat_src, indices, feat_dst, reduction):
    N = indices.shape[0]
    align = 32 if N >= 32 or N == 0 else N
    inputs = [feat_src, indices, feat_dst]

    gnn_utils_cpp.scatter_features(align, reduction, inputs)


def mapped_spmm_copy_lhs_add(dest, indptr, dind, sind, comms, source, edge, soff):
    if edge is None:
        inputs = [dest, indptr, dind, sind, comms, source]
    else:
        inputs = [dest, indptr, dind, sind, comms, source, edge]

    gnn_utils_cpp.mapped_spmm_copy_lhs_add(inputs, rank, soff)

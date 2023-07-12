/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

#include <ATen/record_function.h>
#include <torch/extension.h>
#include <cstdlib>

#ifdef __x86_64__
#include <immintrin.h>
#endif

#include <omp.h>
#include <sched.h>
#include <iostream>
#include <mutex>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
REGISTER_SCOPE(gather, "gather");
REGISTER_SCOPE(scatter, "scatter");

void lfsr_Xwide(unsigned int* rng_state, unsigned int* prng_out, int width) {
  int res = 16 - width;
#ifdef __AVX512F__
  __mmask16 msk = res ? (1 << res) - 1 : 255;
  __m512i vrng_s0 = _mm512_loadu_si512(rng_state);
  __m512i vrng_s1 = _mm512_loadu_si512(rng_state + 16);
  __m512i vrng_s2 = _mm512_loadu_si512(rng_state + 32);
  __m512i vrng_s3 = _mm512_loadu_si512(rng_state + 48);

  __m512i vrng = _mm512_add_epi32(vrng_s3, vrng_s0);
  _mm512_mask_storeu_epi32(prng_out, msk, vrng);

  __m512i vtmp0 = _mm512_slli_epi32(vrng_s1, 9);
  vrng_s2 = _mm512_xor_epi32(vrng_s2, vrng_s0);
  vrng_s3 = _mm512_xor_epi32(vrng_s3, vrng_s1);
  vrng_s1 = _mm512_xor_epi32(vrng_s1, vrng_s2);
  vrng_s0 = _mm512_xor_epi32(vrng_s0, vrng_s3);
  vrng_s2 = _mm512_xor_epi32(vrng_s2, vtmp0);
  vtmp0 = _mm512_slli_epi32(vrng_s3, 11);
  __m512i vtmp1 = _mm512_srli_epi32(vrng_s3, 21);
  vrng_s3 = _mm512_or_epi32(vtmp0, vtmp1);
  _mm512_storeu_si512(rng_state, vrng_s0);
  _mm512_storeu_si512(rng_state + 16, vrng_s1);
  _mm512_storeu_si512(rng_state + 32, vrng_s2);
  _mm512_storeu_si512(rng_state + 48, vrng_s3);
#else
  const unsigned int state_ld = 16;
  int w = res > 0 ? res : 16;
  for (int i = 0; i < w; i++) {
    auto s0 = rng_state[i + (0 * state_ld)];
    auto s1 = rng_state[i + (1 * state_ld)];
    auto s2 = rng_state[i + (2 * state_ld)];
    auto s3 = rng_state[i + (3 * state_ld)];

    unsigned int tmp_0, tmp_1;
    tmp_1 = s3 + s0;
    prng_out[i] = tmp_1;
    tmp_0 = s1 << 9;
    s2 = s2 ^ s0;
    s3 = s3 ^ s1;
    s1 = s1 ^ s2;
    s0 = s0 ^ s3;
    s2 = s2 ^ tmp_0;
    tmp_0 = s3 << 11;
    tmp_1 = s3 >> 21;
    s3 = tmp_0 | tmp_1;
    rng_state[i + (0 * state_ld)] = s0;
    rng_state[i + (1 * state_ld)] = s1;
    rng_state[i + (2 * state_ld)] = s2;
    rng_state[i + (3 * state_ld)] = s3;
  }
#endif
}

at::Tensor gather_features(const long alignN, std::vector<at::Tensor> inputs) {
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "gather.h"
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "gather.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

void scatter_features(
    const long alignN,
    int reduction,
    std::vector<at::Tensor> inputs) {
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
    if (reduction) {
#include "scatter_reduce.h"
    } else {
#include "scatter.h"
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
    if (reduction) {
#include "scatter_reduce.h"
    } else {
#include "scatter.h"
    }
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

std::vector<at::Tensor> find_nodes(
    std::vector<at::Tensor> inputs,
    std::string ntype) {
  auto t_pnd_s_in = inputs[0];
  auto t_pnd_orig = inputs[1];
  auto t_srcnodes = inputs[2];
  auto t_lnodes = inputs[3];

  typedef long Ta;
  typedef int Ts;
#include "find_nodes.h"
}

std::vector<at::Tensor> db_r2l_map(std::vector<at::Tensor> inputs) {
  auto t_db = inputs[0];
  auto t_sn_orig = inputs[1];
  auto t_sn_batch = inputs[2];
  auto t_sn_part = inputs[3];

#include "db_r2l_map.h"
}

std::vector<at::Tensor> r2l_map(std::vector<at::Tensor> inputs) {
  auto t_o2l_map = inputs[0];
  auto t_rbn_orig = inputs[1];

#include "r2l_map.h"
}

std::vector<at::Tensor> find_n_map_nodes(std::vector<at::Tensor> inputs) {
  auto t_db = inputs[0];
  auto t_pnd_solid = inputs[1];
  auto t_pnd_orig = inputs[2];
  auto t_srcnodes = inputs[3];
  auto t_lnodes = inputs[4];

  typedef long Ta;
  typedef int Ts;
#include "find_n_map_solid_nodes.h"
}

void cache_store_n(
    int N,
    int cp,
    long* hmap,
    int* rptr,
    long* nodes,
    int* age,
    int rval,
    int hval,
    at::Tensor t_in_f,
    at::Tensor t_out_f) {
  auto t_loc = at::empty({N}, at::kLong);
  long* loc = t_loc.data_ptr<long>();
#pragma omp parallel for
  for (int j = 0; j < N; j++)
    loc[j] = cp + j;

  int n = 0;
#pragma omp parallel for lastprivate(n)
  for (n = 0; n < ALIGNDOWN(N, 16); n += 16) {
#pragma omp simd
    for (int v = 0; v < 16; v++) {
      if (rptr[loc[n + v]] != rval)
        hmap[n + v] = hval;
      rptr[loc[n + v]] = (int)nodes[n + v];
      age[loc[n + v]] = 0;
      hmap[nodes[n + v]] = loc[n + v];
    }
  }
  if (n < N) {
    int rem = N - n;
    for (int r = 0; r < rem; r++) {
      if (rptr[loc[n + r]] != rval)
        hmap[n + r] = hval;
      rptr[loc[n + r]] = (int)nodes[n + r];
      age[loc[n + r]] = 0;
      hmap[nodes[n + r]] = loc[n + r];
    }
  }
  std::vector<at::Tensor> in;
  in.push_back(t_in_f);
  in.push_back(t_loc);
  in.push_back(t_out_f);
  long alignN = N > 32 ? 32 : N;
  if (N > 0 && alignN > 0)
    scatter_features(alignN, 0, in);
}

void cache_store(
    std::vector<at::Tensor> inputs,
    int cache_size,
    int hval,
    int rval) {
  auto t_hmap = inputs[0];
  auto t_rptr = inputs[1];
  auto t_age = inputs[2];
  auto t_nodes = inputs[3];
  auto t_st_feats = inputs[4];
  auto t_feats = inputs[5];
  auto t_sz_feats = inputs[6];
  auto t_feats_sz = inputs[7];
  auto t_cache_p = inputs[8];

  int* rptr = t_rptr.data_ptr<int>();
  long* hmap = t_hmap.data_ptr<long>();
  long* nodes = t_nodes.data_ptr<long>();
  int* age = t_age.data_ptr<int>();
  int* cp = t_cache_p.data_ptr<int>();

  auto N = t_nodes.size(0);
  if (N > 0) {
    int n = 0;
#pragma omp parallel for lastprivate(n)
    for (n = 0; n < ALIGNDOWN(N, 16); n += 16) {
#pragma omp simd
      for (int v = 0; v < 16; v++) {
        if (hmap[nodes[n + v]] != hval)
          rptr[n + v] = rval;
      }
    }
    if (n < N) {
      int rem = N - n;
      for (int r = 0; r < rem; r++) {
        if (hmap[nodes[n + r]] != hval)
          rptr[n + r] = rval;
      }
    }
    int size = cache_size - cp[0];
    if (size >= N) {
      cache_store_n(
          N, cp[0], hmap, rptr, nodes, age, rval, hval, t_feats, t_st_feats);
      cp[0] += N;
    } else {
      cache_store_n(
          size,
          cp[0],
          hmap,
          rptr,
          nodes,
          age,
          rval,
          hval,
          t_sz_feats,
          t_st_feats);
      int rem = N - size;
      cache_store_n(
          rem,
          0,
          hmap,
          rptr,
          nodes + size,
          age,
          rval,
          hval,
          t_feats_sz,
          t_st_feats);
      cp[0] = rem;
    }
  }
}

std::vector<at::Tensor> cache_load(
    std::vector<at::Tensor> inputs,
    int level,
    int minlife,
    int life) {
  auto t_hmap = inputs[0];
  auto t_oid = inputs[1];
  auto t_feats = inputs[2];
  at::Tensor t_age = at::empty(0, at::kInt);
  if (level > 0)
    t_age = inputs[3];

#include "cache_load.h"
}

void gather_n_store_offset(
    std::vector<at::Tensor> inputs,
    long offseti,
    long offsetv) {
  auto t_in = inputs[0];
  auto t_ind = inputs[1];
  auto t_out = inputs[2];

  auto N = t_ind.size(0);
  long* in = t_in.data_ptr<long>();
  long* ind = t_ind.data_ptr<long>();
  long* out = t_out.data_ptr<long>();

  ind += offseti;
  out += offsetv;

  int n = 0;
#pragma omp parallel for lastprivate(n)
  for (n = 0; n < ALIGNDOWN(N, 16); n += 16) {
#pragma omp simd
    for (int v = 0; v < 16; v++) {
      out[n + v] = in[ind[n + v]];
    }
  }
  if (n < N) {
    int rem = N - n;
    for (int r = 0; r < rem; r++) {
      out[n + r] = in[ind[n + r]];
    }
  }
}

std::vector<at::Tensor> node_sampling(
    std::vector<at::Tensor> inputs,
    const int hil,
    const int thres) {
  auto t_deg = inputs[0];
  auto t_xnbn = inputs[1];
  auto t_xrbn = inputs[2];

  if (t_deg.dtype() == at::kLong) {
    typedef long T;
#include "node_sampling.h"
  } else if (t_deg.dtype() == at::kInt) {
    typedef int T;
#include "node_sampling.h"
  } else if (t_deg.dtype() == at::kFloat) {
    typedef float T;
#include "node_sampling.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

void mapped_spmm_copy_lhs_add(std::vector<at::Tensor> inputs, long soff) {
  auto t_dest = inputs[0];
  auto t_indptr = inputs[1];
  auto t_dind = inputs[2];
  auto t_sind = inputs[3];
  auto t_comms = inputs[4];
  auto t_source = inputs[5];
  auto t_edge = at::empty(0, t_source.options());
  if (inputs.size() == 7)
    t_edge = inputs[6];

  long* indptr = t_indptr.data_ptr<long>();
  long* sind = t_sind.data_ptr<long>();
  long* dind = t_dind.data_ptr<long>();
  int* comms = t_comms.data_ptr<int>();

  auto N = t_indptr.size(0);
  const long F = t_dest.size(1);

  if (t_source.dtype() == at::kFloat) {
    typedef float T;
    auto source = GetVLAPtr<T>(t_source, {F});
    auto dest = GetVLAPtr<T>(t_dest, {F});

    auto add_tpp = AddTPP<T, T>(1, F);

#pragma omp parallel for
    for (int i = 0; i < N - 1; i++) {
      long st = indptr[i];
      long ed = indptr[i + 1];
      for (long j = st; j < ed; j++) {
        int s = comms[sind[j]];
        long d = dind[j];
        if (s == -100)
          continue;

        add_tpp(source[soff + s], dest[d], dest[d]);
      }
    }
  } else if (t_source.dtype() == at::kBFloat16) {
    typedef bfloat16 T;
    auto source = GetVLAPtr<T>(t_source, {F});
    auto dest = GetVLAPtr<T>(t_dest, {F});

    auto add_tpp = AddTPP<T, T>(1, F);

#pragma omp parallel for
    for (int i = 0; i < N - 1; i++) {
      long st = indptr[i];
      long ed = indptr[i + 1];
      for (long j = st; j < ed; j++) {
        int s = comms[sind[j]];
        long d = dind[j];
        if (s == -100)
          continue;
        add_tpp(source[s], dest[d], dest[d]);
      }
    }
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

void affinitize_cores(const int nthreads, const int num_workers) {
#pragma omp parallel
  {
    int mytid = omp_get_thread_num();
    cpu_set_t my_set;
    CPU_ZERO(&my_set);
    CPU_SET(num_workers + mytid, &my_set);

    sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
#ifdef DEBUG
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
      perror("sched_getaffinity");
      assert(false);
    }
    long nproc = nthreads + num_workers;
    for (long i = 0; i < nproc; i++) {
      if (CPU_ISSET(i, &mask))
        printf("%d on core %ld\n", mytid, i);
    }
#endif
  }
}

REGISTER_SUBMODULE(_gnn_utils, m) {
  m.def("gather_features", &gather_features, "C++ Impl of feature gather");
  m.def("scatter_features", &scatter_features, "C++ Impl of feature scatter");
  m.def(
      "find_nodes",
      &find_nodes,
      "C++ Impl of func to gather halo & solid nodes");
  m.def(
      "db_r2l_map",
      &db_r2l_map,
      "C++ Impl of func to gather solid node mapping");
  m.def(
      "r2l_map",
      &r2l_map,
      "C++ Impl of func to find xn of local bn (oid) w/ recv bn (oid)");
  m.def(
      "find_n_map_nodes",
      &find_n_map_nodes,
      "C++ Impl of func to gather solid nodes and get mapping");
  m.def(
      "cache_load",
      &cache_load,
      "C++ impl of func to gather features from cache");
  m.def(
      "cache_store",
      &cache_store,
      "C++ impl of func to scatter features to cache");
  m.def("node_sampling", &node_sampling, "C++ impl of func to sample nodes");
  m.def(
      "gather_n_store_offset",
      &gather_n_store_offset,
      "Gather and store long ints");
  m.def(
      "mapped_spmm_copy_lhs_add",
      &mapped_spmm_copy_lhs_add,
      "C++ impl of gspmm on halo feature graph (drpa)");
  m.def("affinitize_cores", &affinitize_cores, "Compute thread affinization");
}

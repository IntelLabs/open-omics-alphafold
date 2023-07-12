/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

{
  auto N = t_lnodes.size(0);

  Ts* pnd_s_in = t_pnd_s_in.data_ptr<Ts>();
  Ta* pnd_orig = t_pnd_orig.data_ptr<Ta>();
  Ta* src_nodes = t_srcnodes.data_ptr<Ta>();
  Ta* lnodes = t_lnodes.data_ptr<Ta>();

  int threads = omp_get_max_threads();
  std::vector<std::vector<Ta>> idx_thd(threads);

  int check = -1;
  if (ntype == "solid")
    check = 1;
  else if (ntype == "halo")
    check = 0;

  int n;
#pragma omp parallel for lastprivate(n)
  for (n = 0; n < ALIGNDOWN(N, 16); n += 16) {
    int tid = omp_get_thread_num();
#pragma omp simd
    for (int v = 0; v < 16; v++) {
      if (pnd_s_in[lnodes[n + v]] == check)
        idx_thd[tid].push_back(n + v);
    }
  }
  if (n < N) {
    int rem = N - n;
    for (int i = 0; i < rem; i++) {
      if (pnd_s_in[lnodes[n + i]] == check)
        idx_thd[0].push_back(n + i);
    }
  }

  int lN = 0;
  for (int i = 0; i < threads; i++)
    lN += idx_thd[i].size();

  std::vector<Ta> idx(lN);
  long k = 0;

  assert(lN > 0);

  for (int i = 0; i < threads; i++) {
    int n = idx_thd[i].size();
#pragma omp parallel for
    for (int j = 0; j < n; j++)
      idx[k + j] = idx_thd[i][j];
    k += n;
  }

  Ta* idx_ptr = (Ta*)idx.data();

  auto t_orig = t_lnodes.new_empty({lN});
  auto t_batch = t_lnodes.new_empty({lN});
  auto t_part = t_lnodes.new_empty({lN});

  Ta* orig = t_orig.data_ptr<Ta>();
  Ta* batch = t_batch.data_ptr<Ta>();
  Ta* part = t_part.data_ptr<Ta>();

#pragma omp parallel for lastprivate(n)
  for (n = 0; n < ALIGNDOWN(lN, 16); n += 16) {
#pragma omp simd
    for (int v = 0; v < 16; v++) {
      part[n + v] = lnodes[idx_ptr[n + v]];
      batch[n + v] = src_nodes[idx_ptr[n + v]];
      orig[n + v] = pnd_orig[lnodes[idx_ptr[n + v]]];
    }
  }
  if (n < lN) {
    int rem = lN - n;
    for (int i = 0; i < rem; i++) {
      part[n + i] = lnodes[idx_ptr[n + i]];
      batch[n + i] = src_nodes[idx_ptr[n + i]];
      orig[n + i] = pnd_orig[lnodes[idx_ptr[n + i]]];
    }
  }
  return {t_orig, t_batch, t_part};
}

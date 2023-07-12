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

  Ts* db = t_db.data_ptr<Ts>();
  Ts* pnd_solid = t_pnd_solid.data_ptr<Ts>();
  Ta* pnd_orig = t_pnd_orig.data_ptr<Ta>();
  Ta* src_nodes = t_srcnodes.data_ptr<Ta>();
  Ta* lnodes = t_lnodes.data_ptr<Ta>();

  int threads = omp_get_max_threads();
  std::vector<std::vector<Ta>> idx_thd1(threads);

  int n;
#pragma omp parallel for lastprivate(n)
  for (n = 0; n < ALIGNDOWN(N, 1); n += 1) {
    int tid = omp_get_thread_num();
    if (pnd_solid[lnodes[n]] == 1)
      if (db[pnd_orig[lnodes[n]]] == 1)
        idx_thd1[tid].push_back(n);
  }
  if (n < N) {
    int rem = N - n;
    for (int r = 0; r < rem; r++) {
      if (pnd_solid[lnodes[n + r]] == 1)
        if (db[pnd_orig[lnodes[n + r]]] == 1)
          idx_thd1[0].push_back(n + r);
    }
  }

  int lN1 = 0;
  for (int i = 0; i < threads; i++) {
    lN1 += idx_thd1[i].size();
  }

  auto t_idx1 = t_lnodes.new_empty({lN1});
  Ta* idx1 = t_idx1.data_ptr<Ta>();

  long k1 = 0;

  for (int i = 0; i < threads; i++) {
    int n1 = idx_thd1[i].size();
#pragma omp parallel for
    for (int j = 0; j < n1; j++)
      idx1[k1 + j] = idx_thd1[i][j];
    k1 += n1;
  }

  auto t_org = t_lnodes.new_empty({lN1});
  auto t_bat = t_lnodes.new_empty({lN1});
  auto t_par = t_lnodes.new_empty({lN1});

  Ta* org = t_org.data_ptr<Ta>();
  Ta* bat = t_bat.data_ptr<Ta>();
  Ta* par = t_par.data_ptr<Ta>();

#pragma omp parallel for lastprivate(n)
  for (n = 0; n < ALIGNDOWN(lN1, 16); n += 16) {
#pragma omp simd
    for (int v = 0; v < 16; v++) {
      par[n + v] = lnodes[idx1[n + v]];
      bat[n + v] = src_nodes[idx1[n + v]];
      org[n + v] = pnd_orig[lnodes[idx1[n + v]]];
    }
  }
  if (n < lN1) {
    int rem = lN1 - n;
    for (int r = 0; r < rem; r++) {
      par[n + r] = lnodes[idx1[n + r]];
      bat[n + r] = src_nodes[idx1[n + r]];
      org[n + r] = pnd_orig[lnodes[idx1[n + r]]];
    }
  }
  return {t_org, t_bat, t_par};
}

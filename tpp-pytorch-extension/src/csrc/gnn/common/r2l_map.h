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
  auto N = t_rbn_orig.size(0);

  long* o2l_map = t_o2l_map.data_ptr<long>();
  long* rbn_orig = t_rbn_orig.data_ptr<long>();

  int threads = omp_get_max_threads();
  std::vector<std::vector<long>> idx_thd(threads);

  int n;
#pragma omp parallel for lastprivate(n)
  for (n = 0; n < ALIGNDOWN(N, 16); n += 16) {
    int tid = omp_get_thread_num();
#pragma omp simd
    for (int v = 0; v < 16; v++)
      if (o2l_map[rbn_orig[n + v]] != -100)
        idx_thd[tid].push_back(n + v);
  }
  if (n < N) {
    int rem = N - n;
    for (int r = 0; r < rem; r++)
      if (o2l_map[rbn_orig[n + r]] != -100)
        idx_thd[0].push_back(n + r);
  }

  long lN = 0;
  for (int i = 0; i < threads; i++)
    if (idx_thd[i].size() > 0)
      lN += idx_thd[i].size();

  auto t_rbn = t_rbn_orig.new_empty({lN});
  auto t_lid2 = t_rbn_orig.new_empty({lN});

  long* rbn = t_rbn.data_ptr<long>();
  long* lid2 = t_lid2.data_ptr<long>();

  if (lN > 0) {
    std::vector<long> idx(lN);
    unsigned long k = 0;

    for (int i = 0; i < threads; i++) {
      unsigned long n = idx_thd[i].size();
#pragma omp parallel for
      for (unsigned long j = 0; j < n; j++)
        idx[k + j] = idx_thd[i][j];
      k += n;
    }

    long* idx_ptr = (long*)idx.data();
    int n;
#pragma omp parallel for lastprivate(n)
    for (n = 0; n < ALIGNDOWN(lN, 16); n += 16) {
#pragma omp simd
      for (int v = 0; v < 16; v++) {
        rbn[n + v] = rbn_orig[idx_ptr[n + v]];
        lid2[n + v] = o2l_map[rbn_orig[idx_ptr[n + v]]];
      }
    }
    if (n < lN) {
      int rem = lN - n;
      for (int r = 0; r < rem; r++) {
        rbn[n + r] = rbn_orig[idx_ptr[n + r]];
        lid2[n + r] = o2l_map[rbn_orig[idx_ptr[n + r]]];
      }
    }
  }

  return {t_rbn, t_lid2};
}

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
  auto N = t_sn_orig.size(0);

  int* db = t_db.data_ptr<int>();
  long* sn_orig = t_sn_orig.data_ptr<long>();
  long* sn_batch = t_sn_batch.data_ptr<long>();
  long* sn_part = t_sn_part.data_ptr<long>();

  int threads = omp_get_max_threads();
  std::vector<std::vector<long>> idx_thd(threads);

  int n;
#pragma omp parallel for lastprivate(n)
  for (n = 0; n < ALIGNDOWN(N, 16); n += 16) {
    int tid = omp_get_thread_num();
#pragma omp simd
    for (int v = 0; v < 16; v++)
      if (db[sn_orig[n + v]])
        idx_thd[tid].push_back(n + v);
  }
  if (n < N) {
    int rem = N - n;
    for (int r = 0; r < rem; r++)
      if (db[sn_orig[n + r]])
        idx_thd[0].push_back(n + r);
  }

  long lN = 0;
  for (int i = 0; i < threads; i++)
    if (idx_thd[i].size() > 0)
      lN += idx_thd[i].size();

  auto t_orig = t_sn_orig.new_empty({lN});
  auto t_batch = t_sn_batch.new_empty({lN});
  auto t_part = t_sn_part.new_empty({lN});

  long* orig = t_orig.data_ptr<long>();
  long* batch = t_batch.data_ptr<long>();
  long* part = t_part.data_ptr<long>();

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
        orig[n + v] = sn_orig[idx_ptr[n + v]];
        batch[n + v] = sn_batch[idx_ptr[n + v]];
        part[n + v] = sn_part[idx_ptr[n + v]];
      }
    }
    if (n < lN) {
      int rem = lN - n;
      for (int r = 0; r < rem; r++) {
        orig[n + r] = sn_orig[idx_ptr[n + r]];
        batch[n + r] = sn_batch[idx_ptr[n + r]];
        part[n + r] = sn_part[idx_ptr[n + r]];
      }
    }
  }

  return {t_orig, t_batch, t_part};
}

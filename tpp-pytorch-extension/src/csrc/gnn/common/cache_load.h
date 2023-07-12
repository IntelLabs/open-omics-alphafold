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
  auto N = t_oid.size(0);
  long* hmap = t_hmap.data_ptr<long>();
  long* oid = t_oid.data_ptr<long>();
  auto alignN = t_feats.size(0) > 32 ? 32 : t_feats.size(0);

  int threads = omp_get_max_threads();
  std::vector<std::vector<long>> inda_thd(threads);

  int n = 0;
#pragma omp parallel for lastprivate(n)
  for (n = 0; n < ALIGNDOWN(N, 16); n += 16) {
    int tid = omp_get_thread_num();
#pragma omp simd
    for (int v = 0; v < 16; v++) {
      if (hmap[oid[n + v]] != -200)
        inda_thd[tid].push_back(n + v);
    }
  }
  if (n < N) {
    int rem = N - n;
    for (int r = 0; r < rem; r++)
      if (hmap[oid[n + r]] != -200)
        inda_thd[0].push_back(n + r);
  }

  int iN = 0;
  for (int i = 0; i < threads; i++)
    iN += inda_thd[i].size();

  if (iN > 0) {
    auto t_inda = t_oid.new_empty({iN});
    auto t_bitval = t_oid.new_empty({iN});

    long* inda = t_inda.data_ptr<long>();
    long* bv = t_bitval.data_ptr<long>();

    int k = 0;
    for (int i = 0; i < threads; i++) {
      int s = inda_thd[i].size();
#pragma omp parallel for
      for (int j = 0; j < s; j++)
        inda[k + j] = inda_thd[i][j];
      k += s;
    }

    n = 0;
#pragma omp parallel for lastprivate(n)
    for (n = 0; n < ALIGNDOWN(iN, 16); n += 16) {
#pragma omp simd
      for (int v = 0; v < 16; v++) {
        bv[n + v] = hmap[oid[inda[n + v]]];
      }
    }
    if (n < iN) {
      int rem = iN - n;
      for (int r = 0; r < rem; r++)
        bv[n + r] = hmap[oid[inda[n + r]]];
    }

    if (level == 0) {
      std::vector<at::Tensor> in;
      in.push_back(t_feats);
      in.push_back(t_bitval);
      auto t_out = gather_features(alignN, in);
      return {t_inda, t_out};
    } else {
      std::vector<std::vector<long>> indb_thd(threads);
      int* age = t_age.data_ptr<int>();
      n = 0;
#pragma omp parallel for lastprivate(n)
      for (n = 0; n < ALIGNDOWN(iN, 16); n += 16) {
        int tid = omp_get_thread_num();
#pragma omp simd
        for (int v = 0; v < 16; v++) {
          int a = age[bv[n + v]];
          if (a >= minlife && a <= life)
            indb_thd[tid].push_back(n + v);
        }
      }
      if (n < iN) {
        int rem = iN - n;
        for (int r = 0; r < rem; r++) {
          int a = age[bv[n + r]];
          if (a >= minlife && a <= life)
            indb_thd[0].push_back(n + r);
        }
      }
      int bN = 0;
      for (int i = 0; i < threads; i++)
        bN += indb_thd[i].size();

      if (bN > 0) {
        auto t_indb = t_oid.new_empty({bN});
        auto t_lookup_loc = t_oid.new_empty({bN});
        auto t_oid_idx = t_oid.new_empty({bN});

        long* indb = t_indb.data_ptr<long>();
        long* lu = t_lookup_loc.data_ptr<long>();
        long* oidx = t_oid_idx.data_ptr<long>();

        int k = 0;
        for (int i = 0; i < threads; i++) {
          int s = indb_thd[i].size();
#pragma omp parallel for
          for (int j = 0; j < s; j++)
            indb[k + j] = indb_thd[i][j];
          k += s;
        }

        if (bN >= threads / 2) {
          n = 0;
#pragma omp parallel for lastprivate(n)
          for (n = 0; n < ALIGNDOWN(bN, 16); n += 16) {
#pragma omp simd
            for (int v = 0; v < 16; v++) {
              lu[n + v] = bv[indb[n + v]];
              oidx[n + v] = inda[indb[n + v]];
            }
          }
          if (n < bN) {
            int rem = bN - n;
            for (int r = 0; r < rem; r++) {
              lu[n + r] = bv[indb[n + r]];
              oidx[n + r] = inda[indb[n + r]];
            }
          }

          std::vector<at::Tensor> in;
          in.push_back(t_feats);
          in.push_back(t_lookup_loc);
          auto t_out = gather_features(alignN, in);

          return {t_oid_idx, t_out};
        } else {
          for (int n = 0; n < bN; n++) {
            lu[n] = bv[indb[n]];
            oidx[n] = inda[indb[n]];
          }
#if 1
          long fs = t_feats.size(1);
          auto t_out = t_feats.new_empty({bN, fs});
          if (t_feats.dtype() == at::kFloat) {
            auto feats = GetVLAPtr<float>(t_feats, {fs});
            auto out = GetVLAPtr<float>(t_out, {fs});
            for (int n = 0; n < bN; n++) {
#pragma omp parallel for
              for (int f = 0; f < fs; f++) {
                out[n][f] = feats[lu[n]][f];
              }
            }
          } else if (t_feats.dtype() == at::kBFloat16) {
            auto feats = GetVLAPtr<bfloat16>(t_feats, {fs});
            auto out = GetVLAPtr<bfloat16>(t_out, {fs});
            for (int n = 0; n < bN; n++) {
#pragma omp parallel for
              for (int f = 0; f < fs; f++) {
                out[n][f] = feats[lu[n]][f];
              }
            }
          }
#else
          std::vector<at::Tensor> in;
          in.push_back(t_feats);
          in.push_back(t_lookup_loc);
          auto t_out = gather_features(bN, in);
#endif
          return {t_oid_idx, t_out};
        }
      } else {
        auto t_oid_idx = at::empty(0, at::kLong);
        auto t_out = at::empty(0, t_feats.dtype());
        return {t_oid_idx, t_out};
      }
    }
  } else {
    auto t_oid_idx = at::empty(0, at::kLong);
    auto t_out = at::empty(0, t_feats.dtype());
    return {t_oid_idx, t_out};
  }
}

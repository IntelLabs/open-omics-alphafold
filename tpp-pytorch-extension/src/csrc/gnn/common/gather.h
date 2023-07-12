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
  at::Tensor t_in = inputs[0];
  at::Tensor t_idx = inputs[1];

  auto E = t_in.size(1);
  auto N = t_idx.size(0);
  auto nn = N / alignN;
  auto bn = alignN;
  auto rem = N % alignN;

  auto t_out = t_in.new_empty({N, E});

  auto in = GetVLAPtr<T>(t_in, {E});
  auto out = GetVLAPtr<T>(t_out, {bn, E});
  auto idx = GetVLAPtr<int64_t>(t_idx, {bn});

  auto gather_tpp =
      SCOPEIT((EmbeddingFwdTPP<T, int64_t, T>(bn, E, E, E)), ROW_GT);

  {
    RECORD_SCOPE(gather, {t_in, t_idx});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < nn; n++) {
        gather_tpp(in[0], idx[n], out[n][0]);
      }
    }
    if (rem > 0) {
      auto idx = GetVLAPtr<int64_t>(t_idx, {1L});
      auto out = GetVLAPtr<T>(t_out, {E});
      auto gather_tpp =
          SCOPEIT((EmbeddingFwdTPP<T, int64_t, T>(rem, E, E, E)), ROW_GT);
      gather_tpp(in[0], idx[nn * bn], out[nn * bn]);
    }
  }

  return t_out;
}

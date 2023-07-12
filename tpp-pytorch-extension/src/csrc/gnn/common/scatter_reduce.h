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
  at::Tensor t_out = inputs[2];

  auto E = t_in.size(1);
  auto N = t_idx.size(0);
  auto nn = N / alignN;
  auto bn = alignN;
  auto rem = N % alignN;

  auto t_temp = t_out.new_empty({alignN, E});
  auto in = GetVLAPtr<T>(t_in, {bn, E});
  auto out = GetVLAPtr<T>(t_out, {E});
  auto temp = GetVLAPtr<T>(t_temp, {E});
  auto idx = GetVLAPtr<long>(t_idx, {bn});

  auto scatter_tpp = SCOPEIT((ScatterTPP<T, long, T>(bn, E, E, E)), ROW_ST);
  auto gather_tpp = SCOPEIT((EmbeddingFwdTPP<T, long, T>(bn, E, E, E)), ROW_GT);
  auto add_tpp = SCOPEIT((AddTPP<T, T>(bn, E)), EW_ADD);
  {
    RECORD_SCOPE(scatter, {t_out, t_idx});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < nn; n++) {
        gather_tpp(out[0], idx[n], temp[0]);
        add_tpp(in[n][0], temp[0], temp[0]);
        scatter_tpp(temp[0], idx[n], out[0]);
      }
    }
    if (rem > 0) {
      auto t_temp_rem = t_out.new_empty({rem, E});
      auto tempr = GetVLAPtr<T>(t_temp_rem, {E});
      auto idx = GetVLAPtr<long>(t_idx, {1L});
      auto in = GetVLAPtr<T>(t_in, {E});
      auto scatter_tpp =
          SCOPEIT((ScatterTPP<T, long, T>(rem, E, E, E)), ROW_ST);
      auto gather_tpp =
          SCOPEIT((EmbeddingFwdTPP<T, long, T>(rem, E, E, E)), ROW_GT);
      auto add_tpp = SCOPEIT((AddTPP<T, T>(rem, E)), EW_ADD);
      gather_tpp(out[0], idx[nn * bn], tempr[0]);
      add_tpp(in[nn * bn], tempr[0], tempr[0]);
      scatter_tpp(tempr[0], idx[nn * bn], out[0]);
    }
  }
}

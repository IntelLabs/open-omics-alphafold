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

  auto in = GetVLAPtr<T>(t_in, {bn, E});
  auto out = GetVLAPtr<T>(t_out, {E});
  auto idx = GetVLAPtr<int64_t>(t_idx, {bn});

  auto scatter_tpp = SCOPEIT((ScatterTPP<T, int64_t, T>(bn, E, E, E)), ROW_ST);

  {
    RECORD_SCOPE(scatter, {t_out, t_idx});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < nn; n++) {
        scatter_tpp(in[n][0], idx[n], out[0]);
      }
    }
    if (rem > 0) {
      auto idx = GetVLAPtr<int64_t>(t_idx, {1L});
      auto in = GetVLAPtr<T>(t_in, {E});
      auto scatter_tpp =
          SCOPEIT((ScatterTPP<T, int64_t, T>(rem, E, E, E)), ROW_ST);
      scatter_tpp(in[nn * bn], idx[nn * bn], out[0]);
    }
  }
}

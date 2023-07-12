/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("dropout_fwd", std::vector<c10::IValue>());

auto t_in = inp;
at::Tensor t_dp_mask;

if (training && p > 0.0) {
  auto dims = t_in.dim();
  auto N = 1;
  for (int i = 0; i < dims; i++)
    N = N * t_in.sizes()[i];

  int dN = (N + 15) / 16;

  t_dp_mask = at::empty(dN, at::kShort);
  auto in = t_in.data_ptr<T>();
  auto dp_mask = t_dp_mask.data_ptr<short>();

  auto dropout_fwd_tpp = SCOPEIT(DropOutFwdTPP<T>(N, p), DROPOUT);
  {
    RECORD_SCOPE(go_dropout, {t_in});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < N; n++)
        dropout_fwd_tpp(
            &in[n], (void*)get_rng_state(), &in[n], &dp_mask[n / 16]);
    }
  }
}
return {t_in, t_dp_mask};

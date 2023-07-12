/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("dropout_bwd", std::vector<c10::IValue>());

int i = 0;
auto t_grad_out = inputs[i++];
auto t_dp_mask = inputs[i++];

if (p > 0.0) {
  auto dims = t_grad_out.dim();
  auto N = 1;
  for (int i = 0; i < dims; i++)
    N = N * t_grad_out.sizes()[i];

  auto grad_out = t_grad_out.data_ptr<T>();
  auto dp_mask = t_dp_mask.data_ptr<short>();

  auto dropout_bwd_tpp = SCOPEIT(DropOutBwdTPP<T>(N, p), DROPOUT);
  {
    RECORD_SCOPE(gdo_dropout, {t_grad_out});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < N; n++)
        dropout_bwd_tpp(&grad_out[n], &grad_out[n], &dp_mask[n / 16]);
    }
  }
}
return t_grad_out;

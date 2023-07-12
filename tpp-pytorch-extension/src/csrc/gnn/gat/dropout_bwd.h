/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("dropout_bwd", std::vector<c10::IValue>());

int i = 0;
auto t_grad_out = inputs[i++];
auto t_dp_mask = inputs[i++];

if (p > 0.0) {
  auto N = t_grad_out.numel();

  auto grad_out = t_grad_out.data_ptr<T>();
  auto dp_mask = t_dp_mask.data_ptr<short>();

  const int BS = 256; // Define the block size

  auto dropout_bwd_tpp = SCOPEIT(DropOutBwdTPP<T>(BS, p), DROPOUT);
  {
    RECORD_SCOPE(gado_dropout, {t_grad_out});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
      long n;
#pragma omp parallel for lastprivate(n)
      for (n = 0; n < ALIGNDOWN(N, BS); n += BS)
        dropout_bwd_tpp(&grad_out[n], &grad_out[n], &dp_mask[n / 16]);

      if (n < N) {
        auto dropout_bwd_tpp = SCOPEIT(DropOutBwdTPP<T>(N - n, p), DROPOUT);
        dropout_bwd_tpp(&grad_out[n], &grad_out[n], &dp_mask[n / 16]);
      }
    }
  }
}
return t_grad_out;

/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("dropout_fwd", std::vector<c10::IValue>());

auto t_in = inp;
at::Tensor t_dp_mask;

auto N = t_in.numel();
int dN = (N + 15) / 16;
t_dp_mask = at::empty({dN}, at::kShort);

if (training && p > 0.0) {
  auto in = t_in.data_ptr<T>();
  auto dp_mask = t_dp_mask.data_ptr<short>();
  const int BS = 256; // Define the block size

  auto dropout_fwd_tpp = SCOPEIT(DropOutFwdTPP<T>(BS, p), DROPOUT);
  {
    RECORD_SCOPE(gao_dropout, {t_in});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
      long n;
/* The omp parallel loop will run for all the blocks of N except the last block
 by using lastprivate() the ALIGNDOWN() it takes care of the blocks for the 1D
 tensor*/
#pragma omp parallel for lastprivate(n)
      for (n = 0; n < ALIGNDOWN(N, BS); n += BS)
        dropout_fwd_tpp(
            &in[n], (void*)get_rng_state(), &in[n], &dp_mask[n / 16]);

      // The reminder part is handled here
      if (n < N) {
        auto dropout_fwd_tpp = SCOPEIT(DropOutFwdTPP<T>(N - n, p), DROPOUT);
        dropout_fwd_tpp(
            &in[n], (void*)get_rng_state(), &in[n], &dp_mask[n / 16]);
      }
    }
  }
}

else {
  t_dp_mask.zero_();
}

return {t_in, t_dp_mask};

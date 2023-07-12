/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("leakyrelu_fwd", std::vector<c10::IValue>());

auto t_in = inp;
at::Tensor t_lrelu_mask;

auto N = t_in.numel();
int dN = (N + 15) / 16;
t_lrelu_mask = at::empty({dN}, at::kShort);
// auto t_out = t_in.new_empty({t_in.sizes()});
auto t_out = at::empty_like(t_in);
auto out = t_out.data_ptr<T>();

auto in = t_in.data_ptr<T>();
auto lrelu_mask = t_lrelu_mask.data_ptr<short>();
const int BS = 256; // Define the block size

auto lrelu_fwd_tpp = SCOPEIT(LeakyReLUFwdTPP<T>(BS, alpha), ACT);
{
  RECORD_SCOPE(go_lrelu, {t_in});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
    long n;
#pragma omp parallel for lastprivate(n)
    for (n = 0; n < ALIGNDOWN(N, BS); n += BS)
      lrelu_fwd_tpp(&in[n], &out[n], &lrelu_mask[n / 16]);
    if (n < N) {
      auto lrelu_fwd_tpp = SCOPEIT(LeakyReLUFwdTPP<T>(N - n, alpha), ACT);
      lrelu_fwd_tpp(&in[n], &out[n], &lrelu_mask[n / 16]);
    }
  }
}

return {t_out, t_lrelu_mask};

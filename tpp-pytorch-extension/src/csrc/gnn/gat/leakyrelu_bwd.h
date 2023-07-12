/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("leakyrelu_bwd", std::vector<c10::IValue>());

int i = 0;
auto t_grad_in = inputs[i++];
auto t_inp = inputs[i++];
auto t_lrelu_mask = inputs[i++];

auto N = t_grad_in.numel();

auto inp = t_inp.data_ptr<T>();
auto grad_in = t_grad_in.data_ptr<T>();
auto lrelu_mask = t_lrelu_mask.data_ptr<short>();

auto t_grad_out = at::empty_like(t_grad_in);
auto grad_out = t_grad_out.data_ptr<T>();

const int BS = 256; // Define the block size

auto lrelu_bwd_tpp = SCOPEIT(LeakyReLUBwdTPP<T>(BS, alpha), ACT);
{
  RECORD_SCOPE(gdo_lrelu, {t_grad_in});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
    long n;
#pragma omp parallel for lastprivate(n)
    for (n = 0; n < ALIGNDOWN(N, BS); n += BS)
      lrelu_bwd_tpp(&grad_in[n], &grad_out[n], &inp[n], &lrelu_mask[n / 16]);
    if (n < N) {
      auto lrelu_bwd_tpp = SCOPEIT(LeakyReLUBwdTPP<T>(N - n, alpha), ACT);
      lrelu_bwd_tpp(&grad_in[n], &grad_out[n], &inp[n], &lrelu_mask[n / 16]);
    }
  }
}

return t_grad_out;

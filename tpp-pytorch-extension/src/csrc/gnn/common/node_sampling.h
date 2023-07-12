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
  // need to finalize on the data type
  T* deg = t_deg.data_ptr<T>();
  long* xnbn_in = t_xnbn.data_ptr<long>();
  long* xrbn_in = t_xrbn.data_ptr<long>();

  int N = t_deg.size(0);
  auto t_out = at::empty({N}, at::kInt);
  int* out = t_out.data_ptr<int>();
  unsigned int mask = 0xFFFFFFFF >> (32 - hil);
  int n;
  int count = 0;
#pragma omp parallel for lastprivate(n) reduction(+ : count)
  for (n = 0; n < ALIGNDOWN(N, 16); n += 16) {
    unsigned int rnds[16];
    lfsr_Xwide((unsigned int*)get_rng_state(), rnds, 16);
#pragma omp simd
    for (int v = 0; v < 16; v++) {
      rnds[v] &= mask;
      if (rnds[v] < (unsigned int)deg[n + v]) {
        out[n + v] = 1;
        count++;
      } else
        out[n + v] = 0;
    }
  }
  if (n < N) {
    int rem = N - n;
    unsigned int rnds[16];
    lfsr_Xwide((unsigned int*)get_rng_state(), rnds, rem);
    for (int r = 0; r < rem; r++) {
      rnds[r] &= mask;
      if (rnds[r] < (unsigned int)deg[n + r]) {
        out[n + r] = 1;
        count++;
      } else
        out[n + r] = 0;
    }
  }

  if (count >= thres)
    count = thres;
  auto t_xnbn_out = at::empty({count}, at::kLong);
  auto t_xrbn_out = at::empty({count}, at::kLong);

  long* xnbn_out = t_xnbn_out.data_ptr<long>();
  long* xrbn_out = t_xrbn_out.data_ptr<long>();

  count = 0;
  for (int n = 0; n < N; n++) {
    if (out[n] == 1) {
      xnbn_out[count] = xnbn_in[n];
      xrbn_out[count] = xrbn_in[n];
      count++;
    }
    if (count >= thres)
      break;
  }

  return {t_xnbn_out, t_xrbn_out};
}

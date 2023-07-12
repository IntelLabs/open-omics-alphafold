/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("gsage_mlp_fwd", std::vector<c10::IValue>());

at::Tensor t_in, t_in_res, t_wt, t_wt_res, t_bias;
int i = 0;

if (res) {
  t_in = inputs[i++];
  t_in_res = inputs[i++];
  t_wt = inputs[i++];
  t_wt_res = inputs[i++];
} else {
  t_in = inputs[i++];
  t_wt = inputs[i++];
  t_in_res = at::empty(0);
  t_wt_res = at::empty(0);
}
t_bias = inputs[i++];

auto in_sizes = t_in.sizes();
auto wt_sizes = t_wt.sizes();
auto N = in_sizes[0];
auto bn = align;
auto nn = N / bn;
auto rem = N % bn;

auto nk = wt_sizes[0];
auto nc = wt_sizes[1];
auto bc = wt_sizes[2];
if (t_wt.dtype() == at::kBFloat16)
  bc = bc * wt_sizes[4];
auto bk = wt_sizes[3];
auto bcp = bc;
auto K = nk * bk;

if (t_wt.dtype() == at::kBFloat16) {
  bcp = bc + bc % 2;
}

auto t_wt_V = wt_tensor_for_fwd(nk, bk, nc, bc, t_wt);

at::Tensor t_wt_res_V = at::empty(0);

if (res) {
  t_wt_res_V = wt_tensor_for_fwd(nk, bk, nc, bc, t_wt_res);
}

auto t_out = t_in.new_empty({N, K});
at::Tensor t_out_f32;
if (t_out.dtype() == at::kFloat)
  t_out_f32 = t_out;
else
  t_out_f32 = at::empty({N, K});

long rd = (bk + 15) / 16;

at::Tensor t_relu_mask = at::empty({N, nk* rd}, at::kShort);
at::Tensor t_dp_mask = at::empty({N, nk* rd}, at::kShort);

auto in = GetVLAPtr<T>(t_in, {bn, nc, bcp});
auto in_res = GetVLAPtr<T>(t_in_res, {bn, nc, bcp});
auto wt_V = GetVLAPtr<T>(t_wt_V, {nc, bcp* bk});
auto wt_res_V = GetVLAPtr<T>(t_wt_res_V, {nc, bcp* bk});
auto bias = GetVLAPtr<float>(t_bias, {bk});
auto out = GetVLAPtr<T>(t_out, {bn, nk, bk});
auto out_f32 = GetVLAPtr<float>(t_out_f32, {bn, nk, bk});
auto relu_mask = GetVLAPtr<short>(t_relu_mask, {bn, nk, rd});
auto dp_mask = GetVLAPtr<short>(t_dp_mask, {bn, nk, rd});

auto brgemm_tpp = SCOPEIT(
    (BrgemmTPP<
        T,
        float>(bn, bk, bcp, bcp, bk* bcp, nc* bcp, bk, nk* bk, 1.0, 0, nc)));
auto cpy_bias_tpp = SCOPEIT(CpyBiasTPP<float>(bn, bk, K), BIAS);
auto relu_fwd_tpp =
    SCOPEIT(ReLUFwdTPP<float>(bn, bk, nk* bk, nk* bk, true), ACT);
auto dropout_fwd_tpp =
    SCOPEIT((DropOutFwdTPP<float, T>(bn, bk, nk* bk, nk* bk, p)), DROPOUT);
auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(bn, bk, K, K)), EW_COPY);

{
  RECORD_SCOPE(go_gemm, {t_in, t_wt_V});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int threads = omp_get_num_threads();
      int work = nn * nk;
      int chunk =
          (work % threads == 0) ? (work / threads) : (work / threads) + 1;
      int chunk_start = (tid * chunk < work) ? (tid * chunk) : work;
      int chunk_end = ((tid + 1) * chunk < work) ? ((tid + 1) * chunk) : work;

      brgemm_tpp.config();

      for (int n3k = chunk_start; n3k < chunk_end; n3k++) {
        int n = n3k / nk;
        int k = n3k % nk;
        if (apply_bias)
          cpy_bias_tpp(bias[k], out_f32[n][0][k]);
        brgemm_tpp(in[n][0][0], wt_V[k][0], out_f32[n][0][k], nc, true);
        if (res) {
          brgemm_tpp(
              in_res[n][0][0], wt_res_V[k][0], out_f32[n][0][k], nc, true);
        }
        if (act == "relu") {
          relu_fwd_tpp(out_f32[n][0][k], out_f32[n][0][k], relu_mask[n][0][k]);
        }
        if (p > 0 && training) {
          dropout_fwd_tpp(
              out_f32[n][0][k],
              (void*)get_rng_state(),
              out[n][0][k],
              dp_mask[n][0][k]);
        } else
          cvt_tpp(out_f32[n][0][k], out[n][0][k]);
      }
      brgemm_tpp.release();
    }
    if (rem > 0) {
      auto in = GetVLAPtr<T>(t_in, {nc, bcp});
      auto in_res = GetVLAPtr<T>(t_in_res, {nc, bcp});
      auto out = GetVLAPtr<T>(t_out, {nk, bk});
      auto out_f32 = GetVLAPtr<float>(t_out_f32, {nk, bk});
      auto relu_mask = GetVLAPtr<short>(t_relu_mask, {nk, rd});
      auto dp_mask = GetVLAPtr<short>(t_dp_mask, {nk, rd});

      auto brgemm_tpp = SCOPEIT((BrgemmTPP<T, float>(
          rem, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 1.0, 0, nc)));
      auto cpy_bias_tpp = SCOPEIT(CpyBiasTPP<float>(1, bk, K), BIAS);
      auto relu_fwd_tpp =
          SCOPEIT(ReLUFwdTPP<float>(1, bk, nk * bk, nk * bk, true), ACT);
      auto dropout_fwd_tpp = SCOPEIT(
          (DropOutFwdTPP<float, T>(1, bk, nk * bk, nk * bk, p)), DROPOUT);
      auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(1, bk, K, K)), EW_COPY);

#pragma omp parallel
      {
        brgemm_tpp.config();
#pragma omp for
        for (int k = 0; k < nk; k++) {
          if (apply_bias)
            for (int r = 0; r < rem; r++)
              cpy_bias_tpp(bias[k], out_f32[nn * bn + r][k]);
          brgemm_tpp(in[nn * bn][0], wt_V[k][0], out_f32[nn * bn][k], nc, true);
          if (res) {
            brgemm_tpp(
                in_res[nn * bn][0],
                wt_res_V[k][0],
                out_f32[nn * bn][k],
                nc,
                true);
          }

          for (int r = 0; r < rem; r++) {
            if (act == "relu") {
              relu_fwd_tpp(
                  out_f32[nn * bn + r][k],
                  out_f32[nn * bn + r][k],
                  relu_mask[nn * bn + r][k]);
            }
            if (p > 0 && training) {
              dropout_fwd_tpp(
                  out_f32[nn * bn + r][k],
                  (void*)get_rng_state(),
                  out[nn * bn + r][k],
                  dp_mask[nn * bn + r][k]);
            } else
              cvt_tpp(out_f32[nn * bn + r][k], out[nn * bn + r][k]);
          }
        }
        brgemm_tpp.release();
      }
    }
  }
}

return {t_out, t_relu_mask, t_dp_mask};

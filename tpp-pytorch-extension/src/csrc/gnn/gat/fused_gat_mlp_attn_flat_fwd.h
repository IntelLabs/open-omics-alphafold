/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("gat_mlp_fwd", std::vector<c10::IValue>());

at::Tensor t_in_mlp, t_attn_3d, t_wt, t_bias;
int i = 0;
// #define PRINT_T(x) std::cout << #x << "==: " << x.sizes() << std::endl

t_in_mlp = inputs[i++]; // [N, C]
t_wt = inputs[i++]; // [nk, nc, bc, bk]
t_attn_3d = inputs[i++]; // [1, H, F]
t_bias = inputs[i++]; // [K]

auto in_sizes = t_in_mlp.sizes();
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

auto t_out_mlp = t_in_mlp.new_empty({N, K}); // [N,  K]
at::Tensor t_out_f32;
if (t_out_mlp.dtype() == at::kFloat)
  t_out_f32 = t_out_mlp;
else
  t_out_f32 = at::empty({N, K});

long rd = (bk + 15) / 16;

at::Tensor t_relu_mask = at::empty({N, nk* rd}, at::kShort);

auto in = GetVLAPtr<T>(t_in_mlp, {bn, nc, bcp});
auto wt_V = GetVLAPtr<T>(t_wt_V, {nc, bcp* bk});
auto bias = GetVLAPtr<float>(t_bias, {bk});
auto out = GetVLAPtr<T>(t_out_mlp, {bn, nk, bk});
auto out_f32 = GetVLAPtr<float>(t_out_f32, {bn, nk, bk});
auto relu_mask = GetVLAPtr<short>(t_relu_mask, {bn, nk, rd});

auto brgemm_tpp = SCOPEITGEMM2(
    (BrgemmTPP<
        T,
        float>(bn, bk, bcp, bcp, bk* bcp, nc* bcp, bk, nk* bk, 1.0, 0, nc)));

auto cpy_bias_tpp = SCOPEIT(CpyBiasTPP<float>(bn, bk, K), BIAS);
auto relu_fwd_tpp =
    SCOPEIT(ReLUFwdTPP<float>(bn, bk, nk* bk, nk* bk, true), ACT);
auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(bn, bk, K, K)), EW_COPY);

{
  RECORD_SCOPE(gao_gemm, {t_in_mlp, t_wt_V});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int n = 0; n < nn; n++) {
      for (int k = 0; k < nk; k++) {
        cpy_bias_tpp(bias[k], out_f32[n][0][k]);
        brgemm_tpp(in[n][0][0], wt_V[k][0], out_f32[n][0][k], nc);
        if (act == "relu") {
          relu_fwd_tpp(out_f32[n][0][k], out_f32[n][0][k], relu_mask[n][0][k]);
        }
        cvt_tpp(out_f32[n][0][k], out[n][0][k]);
      }
    }
    if (rem > 0) {
      auto in = GetVLAPtr<T>(t_in_mlp, {nc, bcp});
      auto out = GetVLAPtr<T>(t_out_mlp, {nk, bk});
      auto out_f32 = GetVLAPtr<float>(t_out_f32, {nk, bk});
      auto relu_mask = GetVLAPtr<short>(t_relu_mask, {nk, rd});

      auto brgemm_tpp = SCOPEITGEMM2((BrgemmTPP<T, float>(
          rem, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 1.0, 0, nc)));

      auto cpy_bias_tpp = SCOPEIT(CpyBiasTPP<float>(1, bk, K), BIAS);
      auto relu_fwd_tpp =
          SCOPEIT(ReLUFwdTPP<float>(1, bk, nk * bk, nk * bk, true), ACT);
      auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(1, bk, K, K)), EW_COPY);

#pragma omp parallel for
      for (int k = 0; k < nk; k++) {
        for (int r = 0; r < rem; r++)
          cpy_bias_tpp(bias[k], out_f32[nn * bn + r][k]);
        brgemm_tpp(in[nn * bn][0], wt_V[k][0], out_f32[nn * bn][k], nc);
        for (int r = 0; r < rem; r++) {
          if (act == "relu") {
            relu_fwd_tpp(
                out_f32[nn * bn + r][k],
                out_f32[nn * bn + r][k],
                relu_mask[nn * bn + r][k]);
          }
          cvt_tpp(out_f32[nn * bn + r][k], out[nn * bn + r][k]);
        }
      }
    }
  }
}

auto attn_sizes = t_attn_3d.sizes(); // 3D shape [1, H, F] = [1, 4, 128] let

auto H = attn_sizes[1]; // 4
auto F = attn_sizes[2]; // 128

auto t_out_attn = t_out_mlp.new_empty({N, H});

auto t_attn = t_attn_3d.view({H * F});

at::Tensor t_out_attn_f32;
if (t_in_mlp.dtype() == at::kBFloat16)
  t_out_attn_f32 = at::empty({N, H});
else
  t_out_attn_f32 = t_out_attn;

auto in_attn = GetVLAPtr<T>(t_out_mlp, {H, F});
auto attn = GetVLAPtr<T>(t_attn, {F}); // nk, bk
auto out_attn = GetVLAPtr<T>(t_out_attn, {H}); // N, H
auto out_attn_f32 = GetVLAPtr<float>(t_out_attn_f32, {H}); // N, H

auto mul_reduce_tpp = SCOPEIT((MulReduceTPP<T, T, float>(H, F)), EW_MUL);
auto cvt_attn_tpp = SCOPEIT((ConvertTPP<float, T>(1, H)), EW_COPY);

{
  RECORD_SCOPE(go_attn, {t_out_attn});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      mul_reduce_tpp(attn[0], in_attn[n][0], out_attn_f32[n]);
    }
    if (t_out_attn.dtype() != t_out_attn_f32.dtype()) {
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        int threads = omp_get_num_threads();
        int work = N;
        int chunk =
            (work % threads == 0) ? (work / threads) : (work / threads) + 1;
        int chunk_start = (tid * chunk < work) ? (tid * chunk) : work;
        int chunk_end = ((tid + 1) * chunk < work) ? ((tid + 1) * chunk) : work;

        for (int r = chunk_start; r < chunk_end; r++)
          cvt_attn_tpp(out_attn_f32[r], out_attn[r]);
      }
    }
  }
}

return {t_out_mlp, t_out_attn.view({N, H, 1}), t_relu_mask};

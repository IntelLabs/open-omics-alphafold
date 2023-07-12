/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("bert_fwd", std::vector<c10::IValue>());
int i = 0;
auto t_in = inputs[i++]; // [B][S1][Nc][S2][Hc]
auto t_in2 = inputs[i++]; // [B][S1][Nk][S2][Hk]
auto t_wt = inputs[i++]; // [Nk][Nc][Hc][Hk]
auto t_bias = inputs[i++]; // [Nk][Hk]
auto t_gamma = inputs[i++]; // [Nk][Hk]
auto t_beta = inputs[i++]; // [Nk][Hk]
auto in_sizes = t_in.sizes();
auto wt_sizes = t_wt.sizes();
auto B = in_sizes[0];
auto S1 = in_sizes[1];
auto Nc = in_sizes[2];
auto S2 = in_sizes[3];
auto Hc = in_sizes[4];

auto Nk = wt_sizes[0];
auto Hk = wt_sizes[3];

auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

auto t_dout = t_in.new_empty({B, S1, Nk, S2, Hk});
auto t_out = t_dout;
if (training) {
  t_out = t_in.new_empty({B, S1, Nk, S2, Hk});
}

// auto t_dp_mask = at::Tensor();
auto t_dp_mask = at::empty({B, S1, Nk, (S2 * Hk + 15) / 16}, at::kShort);
auto t_mean = t_gamma.new_empty({B, S1, S2}, at::kFloat);
auto t_var = t_gamma.new_empty({B, S1, S2}, at::kFloat);

if (p > 0)
  t_dp_mask = at::empty({B, S1, Nk, (S2 * Hk + 15) / 16}, at::kShort);

auto in = GetVLAPtr<T>(t_in, {S1, Nc, S2, Hc});
auto in2 = GetVLAPtr<T>(t_in2, {S1, Nk, S2, Hk});
// auto  wt_V = GetVLAPtr<T>( t_wt_V, { Nc, Hc / 2, Hk, 2});
auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc* Hk});
auto bias = GetVLAPtr<T>(t_bias, {Hk});
auto gamma = GetVLAPtr<T>(t_gamma, {Hk});
auto beta = GetVLAPtr<T>(t_beta, {Hk});
auto mean = GetVLAPtr<float>(t_mean, {S1, S2});
auto var = GetVLAPtr<float>(t_var, {S1, S2});
auto dout = GetVLAPtr<T>(t_dout, {S1, Nk, S2, Hk});
auto out = GetVLAPtr<T>(t_out, {S1, Nk, S2, Hk});
auto dp_mask = GetVLAPtr<short>(t_dp_mask, {S1, Nk, (S2 * Hk + 15) / 16});

auto Ncb = Nc;
if (Nc > Nk && Nc % Nk == 0) {
  Ncb = Nk;
}
// Create TPPs
auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(S2, Hk), BIAS);
auto brgemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
    S2,
    Hk,
    Hc,
    S2* Hc,
    Hk* Hc,
    1.0,
    XformTPP::XFORM_NONE_TPP,
    0,
    Ncb)));
auto dropout_fwd_tpp = SCOPEIT(DropOutFwdTPP<T>(S2 * Hk, p), DROPOUT);
auto add_tpp = SCOPEIT((AddTPP<T, T>(S2 * Hk)), EW_ADD);
auto layer_norm_fwd_tpp =
    SCOPEIT(LayerNormFwdTPP<T>(Nk, S2, Hk, eps), LAYER_NORM);

{
  RECORD_SCOPE(o_gemm, {t_in, t_wt});
  auto nThreads = omp_get_max_threads();
  for (int nc = 0; nc < Nc; nc += Ncb) {
    if (nc == Nc - Ncb) {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
      if (nThreads < B * S1) {
#pragma omp parallel for collapse(2)
        for (int b = 0; b < B; b++) {
          for (int s1 = 0; s1 < S1; s1++) {
            for (int nk = 0; nk < Nk; nk++) {
              if (nc == 0) {
                copy_bias_tpp(bias[nk], dout[b][s1][nk][0]);
              }
              brgemm_tpp(
                  in[b][s1][nc][0], wt_V[nk][nc], dout[b][s1][nk][0], Ncb);
              if (p > 0) {
                dropout_fwd_tpp(
                    dout[b][s1][nk][0],
                    (void*)get_rng_state(),
                    dout[b][s1][nk][0],
                    dp_mask[b][s1][nk]);
              }
              add_tpp(
                  dout[b][s1][nk][0], in2[b][s1][nk][0], dout[b][s1][nk][0]);
            }
            layer_norm_fwd_tpp(
                dout[b][s1][0][0],
                gamma[0],
                beta[0],
                mean[b][s1],
                var[b][s1],
                out[b][s1][0][0]);
          }
        }
      } else {
#pragma omp parallel for collapse(3)
        for (int b = 0; b < B; b++) {
          for (int s1 = 0; s1 < S1; s1++) {
            for (int nk = 0; nk < Nk; nk++) {
              if (nc == 0) {
                copy_bias_tpp(bias[nk], dout[b][s1][nk][0]);
              }
              brgemm_tpp(
                  in[b][s1][nc][0], wt_V[nk][nc], dout[b][s1][nk][0], Ncb);
              if (p > 0) {
                dropout_fwd_tpp(
                    dout[b][s1][nk][0],
                    (void*)get_rng_state(),
                    dout[b][s1][nk][0],
                    dp_mask[b][s1][nk]);
              }
              add_tpp(
                  dout[b][s1][nk][0], in2[b][s1][nk][0], dout[b][s1][nk][0]);
            }
          }
        }
#pragma omp parallel for collapse(2)
        for (int b = 0; b < B; b++) {
          for (int s1 = 0; s1 < S1; s1++) {
            layer_norm_fwd_tpp(
                dout[b][s1][0][0],
                gamma[0],
                beta[0],
                mean[b][s1],
                var[b][s1],
                out[b][s1][0][0]);
          }
        }
      }
    } else {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
      for (int b = 0; b < B; b++) {
        for (int s1 = 0; s1 < S1; s1++) {
          for (int nk = 0; nk < Nk; nk++) {
            if (nc == 0) {
              copy_bias_tpp(bias[nk], dout[b][s1][nk][0]);
            }
            brgemm_tpp(in[b][s1][nc][0], wt_V[nk][nc], dout[b][s1][nk][0], Ncb);
          }
        }
      }
    }
  }
}
return std::vector<at::Tensor>({t_out, t_dout, t_mean, t_var, t_dp_mask});

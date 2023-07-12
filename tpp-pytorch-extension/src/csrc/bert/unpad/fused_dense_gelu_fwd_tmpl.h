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
auto in_sizes = t_in.sizes();
auto wt_sizes = t_wt.sizes();
auto S1 = in_sizes[0];
auto Nc = in_sizes[1];
auto S2 = in_sizes[2];
auto Hc = in_sizes[3];

auto Nk = wt_sizes[0];
auto Hk = wt_sizes[3];

auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);

auto t_gelu_out = t_in.new_empty({S1, Nk, S2, Hk});
auto t_out = t_gelu_out;
if (training) {
  t_out = t_in.new_empty({S1, Nk, S2, Hk});
}

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
auto gelu_fwd_tpp = SCOPEIT(GeluFwdTPP<T>(S2 * Hk), ACT);

auto in = GetVLAPtr<T>(t_in, {Nc, S2* Hc});
auto wt_V = GetVLAPtr<T>(t_wt_V, {Nc, Hc* Hk});
auto bias = GetVLAPtr<T>(t_bias, {Hk});
auto out = GetVLAPtr<T>(t_out, {Nk, S2* Hk});
auto gelu_out = GetVLAPtr<T>(t_gelu_out, {Nk, S2* Hk});

{
  RECORD_SCOPE(i_gemm, {t_in, t_wt_V});
#ifdef NO_PARLOOPER
  for (int nc = 0; nc < Nc; nc += Ncb) {
#pragma omp parallel for collapse(2)
    for (int s1 = 0; s1 < S1; s1++) {
      for (int nk = 0; nk < Nk; nk++) {
        if (nc == 0) {
          copy_bias_tpp(bias[nk], out[s1][nk]);
        }
        brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], Ncb);
        if (nc == Nc - Ncb) { // last iter
          gelu_fwd_tpp(out[s1][nk], gelu_out[s1][nk]);
        }
      }
    }
  }
#else
  auto loop_scheme = large_cache_opt ? "acB" : "aBC";
  // auto gemm_loop =
  //    ThreadedLoop<3>({{0, Nc, Ncb, false}, {S1}, {Nk}}, loop_scheme);
  auto gemm_loop = ThreadedLoop<3>(
      {LoopSpecs{0, Nc, Ncb, false}, LoopSpecs{S1}, LoopSpecs{Nk}},
      loop_scheme);
  gemm_loop(
      [&](int* ind) {
        int nc = ind[0], s1 = ind[1], nk = ind[2];

        if (nc == 0) {
          copy_bias_tpp(bias[nk], out[s1][nk]);
        }
        brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], Ncb, true);
        if (nc == Nc - Ncb) { // last iter
          gelu_fwd_tpp(out[s1][nk], gelu_out[s1][nk]);
        }
      },
      [&]() { brgemm_tpp.config(); },
      [&]() { brgemm_tpp.release(); });

#endif
}
return std::vector<at::Tensor>({t_out, t_gelu_out});

/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("bert_bwd", std::vector<c10::IValue>());
auto in_sizes = t_in.sizes();
auto wt_sizes = t_wt.sizes();
auto B = in_sizes[0];
auto S1 = in_sizes[1];
auto Nc = in_sizes[2];
auto S2 = in_sizes[3];
auto Hc = in_sizes[4];

auto Nk = wt_sizes[0];
auto Hk = wt_sizes[3];

const auto grad_wt_flag =
    (t_wt.dim() == 5 ? XformTPP::XFORM_N2V_TPP : XformTPP::XFORM_NONE_TPP);
const auto input_trans_flag =
    (t_in.dtype() == at::kFloat ? XformTPP::XFORM_XPOSE_TPP
                                : XformTPP::XFORM_NONE_TPP);
auto t_wt_TV = wt_tensor_for_bwd(Nk, Hk, Nc, Hc, t_wt);

auto t_in_T = t_in;
if (input_trans_flag == XformTPP::XFORM_NONE_TPP) {
  t_in_T = act_tensor_trans(B, S1, Nc, S2, Hc, t_in);
}

auto t_grad_in = at::empty_like(t_in);
auto t_grad_gelu = at::empty_like(t_grad_out);
auto t_grad_wt = at::empty_like(t_wt);
auto t_grad_bias = t_wt.new_empty({Nk * Hk}); // [Nk][Hk]
auto t_grad_gelu_V = t_grad_gelu;
if (t_grad_gelu.dtype() == at::kBFloat16) {
  t_grad_gelu_V = t_grad_out.new_empty({B, S1, Nk, S2 / 2, Hk, 2});
}

auto in_T = GetVLAPtr<T>(t_in_T, {S1, Nc, Hc* S2});
auto gelu_in = GetVLAPtr<T>(t_gelu_in, {S1, Nk, S2* Hk});
auto grad_in = GetVLAPtr<T>(t_grad_in, {S1, Nc, S2* Hc});
auto wt_TV = GetVLAPtr<T>(t_wt_TV, {Nc, Hk* Hc});
auto grad_wt = GetVLAPtr<T>(t_grad_wt, {Nc, Hc* Hk});
auto grad_bias = GetVLAPtr<T>(t_grad_bias, {Hk});
auto grad_gelu = GetVLAPtr<T>(t_grad_gelu, {S1, Nk, S2* Hk});
auto grad_out = GetVLAPtr<T>(t_grad_out, {S1, Nk, S2* Hk});
auto grad_gelu_V = GetVLAPtr<T>(t_grad_gelu_V, {S1, Nk, S2* Hk});

auto Nkb = Nk;
if (Nk > Nc && Nk % Nc == 0) {
  Nkb = Nc;
}

auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(Nk * Hk), EW_ZERO);
auto gelu_bwd_tpp = SCOPEIT(GeluBwdTPP<T>(S2 * Hk), ACT);
auto grad_bias_tpp = SCOPEIT(GradBiasTPP<T>(S2, Hk), BIAS);
auto n2v_tpp =
    SCOPEIT(XformExtTPP<T>(S2, Hk, XformTPP::XFORM_N2V_TPP, true), VNNI);
auto di_gemm_b0_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
    S2,
    Hc,
    Hk,
    S2* Hk,
    Nc* Hk* Hc,
    0.0,
    XformTPP::XFORM_NONE_TPP,
    0,
    Nkb)));
auto di_gemm_b1_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
    S2,
    Hc,
    Hk,
    S2* Hk,
    Nc* Hk* Hc,
    1.0,
    XformTPP::XFORM_NONE_TPP,
    0,
    Nkb)));
auto dw_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
    Hc,
    Hk,
    S2,
    Nc* S2* Hc,
    Nk* S2* Hk,
    1.0,
    (XformTPP::XFORM_TYPE)grad_wt_flag,
    input_trans_flag,
    S1)));
{
  RECORD_SCOPE(di_bias, {t_grad_out});
  // t_grad_bias.zero_();
  tensor_set_zero(Nk, Hk, t_grad_bias);
  int num_threads = omp_get_max_threads();
  float* bias_ptrs[num_threads];
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      float prv_grad_bias[Nk][Hk];
      bias_ptrs[tid] = prv_grad_bias[0];
      set_zero_tpp(prv_grad_bias[0]);
#pragma omp for collapse(3) // reduction(+:grad_bias[:Nk][:Hk])
      for (int b = 0; b < B; b++) {
        for (int s1 = 0; s1 < S1; s1++) {
          for (int nk = 0; nk < Nk; nk++) {
            gelu_bwd_tpp(
                grad_out[b][s1][nk], gelu_in[b][s1][nk], grad_gelu[b][s1][nk]);
            grad_bias_tpp(grad_gelu[b][s1][nk], prv_grad_bias[nk]);
            n2v_tpp(grad_gelu[b][s1][nk], grad_gelu_V[b][s1][nk]);
          }
        }
      }
#pragma omp barrier
      omp_reduce_buf(num_threads, Nk * Hk, bias_ptrs, grad_bias[0]);
    }
  }
}
{
  RECORD_SCOPE(dii_gemm, {t_grad_gelu, t_wt_TV});
  // if(Nk != Nkb) t_grad_in.zero_();
  if (Nk != Nkb)
    tensor_set_zero(B * S1 * Nc, S2 * Hc, t_grad_in);
  for (int nk = 0; nk < Nk; nk += Nkb) {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
      for (int s1 = 0; s1 < S1; s1++) {
        for (int nc = 0; nc < Nc; nc++) {
          if (Nk != Nkb)
            di_gemm_b1_tpp(
                grad_gelu[b][s1][nk], wt_TV[nk][nc], grad_in[b][s1][nc], Nkb);
          else
            di_gemm_b0_tpp(
                grad_gelu[b][s1][nk], wt_TV[nk][nc], grad_in[b][s1][nc], Nkb);
        }
      }
    }
  }
}
{
  RECORD_SCOPE(dwi_gemm, {t_in_T, t_grad_gelu_V});
  // t_grad_wt.zero_();
  tensor_set_zero(Nk * Nc, Hk * Hc, t_grad_wt);
  for (int b = 0; b < B; b++) {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int nk = 0; nk < Nk; nk++) {
      for (int nc = 0; nc < Nc; nc++) {
        dw_gemm_tpp(in_T[b][0][nc], grad_gelu_V[b][0][nk], grad_wt[nk][nc], S1);
      }
    }
  }
}
return std::vector<at::Tensor>({t_grad_in, t_grad_wt, t_grad_bias});

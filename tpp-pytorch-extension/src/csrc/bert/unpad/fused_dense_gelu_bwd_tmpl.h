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
auto S1 = in_sizes[0];
auto Nc = in_sizes[1];
auto S2 = in_sizes[2];
auto Hc = in_sizes[3];

auto Nk = wt_sizes[0];
auto Hk = wt_sizes[3];

const int VBS = get_vnni_block_size<T>();
const auto grad_wt_flag =
    (t_wt.dim() == 5 ? XformTPP::XFORM_N2V_TPP : XformTPP::XFORM_NONE_TPP);
const auto input_trans_flag =
    (t_in.dtype() == at::kFloat ? XformTPP::XFORM_XPOSE_TPP
                                : XformTPP::XFORM_NONE_TPP);
auto t_wt_TV = wt_tensor_for_bwd_compact(Nk, Hk, Nc, Hc, t_wt);

auto t_in_T = t_in;
if (input_trans_flag == XformTPP::XFORM_NONE_TPP) {
  t_in_T = act_tensor_trans_compact(S1, Nc, S2, Hc, t_in);
}
auto in_blk = LToPBlockAccessMapper<T>(S1, Nc);

auto t_grad_in = at::empty_like(t_in);
auto t_grad_gelu = at::empty_like(t_grad_out);
auto t_grad_wt = at::empty_like(t_wt);
auto t_grad_bias = t_wt.new_empty({Nk * Hk}); // [Nk][Hk]
auto t_grad_gelu_V = t_grad_gelu;
if (t_grad_gelu.dtype() != at::kFloat) {
  t_grad_gelu_V = t_grad_out.new_empty({Nk, S1, S2 / VBS, Hk, VBS});
}
auto gdout_blk = LToPBlockAccessMapper<T>(S1, Nk);

auto in_T = GetVLAPtr<T>(t_in_T, {Hc * S2});
auto gelu_in = GetVLAPtr<T>(t_gelu_in, {Nk, S2* Hk});
auto grad_in = GetVLAPtr<T>(t_grad_in, {Nc, S2* Hc});
auto wt_TV = GetVLAPtr<T>(t_wt_TV, {Nk, Hk* Hc});
auto grad_wt = GetVLAPtr<T>(t_grad_wt, {Nc, Hc* Hk});
auto grad_bias = GetVLAPtr<T>(t_grad_bias, {Hk});
auto grad_gelu = GetVLAPtr<T>(t_grad_gelu, {Nk, S2* Hk});
auto grad_out = GetVLAPtr<T>(t_grad_out, {Nk, S2* Hk});
auto grad_gelu_V = GetVLAPtr<T>(t_grad_gelu_V, {S2 * Hk});

const long BS = 8;
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
    Hk* Hc,
    0.0,
    XformTPP::XFORM_NONE_TPP,
    0,
    Nkb)));
auto di_gemm_b1_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
    S2,
    Hc,
    Hk,
    S2* Hk,
    Hk* Hc,
    1.0,
    XformTPP::XFORM_NONE_TPP,
    0,
    Nkb)));
auto dw_set_zero_tpp = SCOPEIT(SetZeroTPP<T>(Hk * Hc), EW_ZERO);
auto dw_cpy_tpp = SCOPEIT(CpyTPP<T>(Hk * Hc), VNNI);
auto dw_n2v_tpp =
    SCOPEIT(XformExtTPP<T>(Hc, Hk, XformTPP::XFORM_N2V_TPP, true), VNNI);
auto dw_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
    Hc,
    Hk,
    S2,
    input_trans_flag == XformTPP::XFORM_NONE_TPP ? S2 * Hc : Nc * S2 * Hc,
    input_trans_flag == XformTPP::XFORM_NONE_TPP ? S2 * Hk : Nk * S2 * Hk,
    1.0,
    XformTPP::XFORM_NONE_TPP, //(XformTPP::XFORM_TYPE)grad_wt_flag,
    input_trans_flag,
    BS)));
{
  RECORD_SCOPE(di_bias, {t_grad_out});
  tensor_set_zero(Nk, Hk, t_grad_bias);
  int num_threads = omp_get_max_threads();
  float* bias_ptrs[num_threads];
  {
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      float prv_grad_bias[Nk][Hk];
      bias_ptrs[tid] = prv_grad_bias[0];
      set_zero_tpp(prv_grad_bias[0]);
#pragma omp for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int nk = 0; nk < Nk; nk++) {
          gelu_bwd_tpp(grad_out[s1][nk], gelu_in[s1][nk], grad_gelu[s1][nk]);
          grad_bias_tpp(grad_gelu[s1][nk], prv_grad_bias[nk]);
          n2v_tpp(grad_gelu[s1][nk], grad_gelu_V[gdout_blk(s1, nk)]);
        }
      }
      omp_reduce_buf(num_threads, Nk * Hk, bias_ptrs, grad_bias[0]);
    }
  }
}
{
  RECORD_SCOPE(dii_gemm, {t_grad_gelu, t_wt_TV});
#ifdef NO_PARLOOPER
  for (int nk = 0; nk < Nk; nk += Nkb) {
#pragma omp parallel for collapse(2)
    for (int s1 = 0; s1 < S1; s1++) {
      for (int nc = 0; nc < Nc; nc++) {
        if (nk == 0)
          di_gemm_b0_tpp(
              grad_gelu[s1][nk], wt_TV[nc][nk], grad_in[s1][nc], Nkb);
        else
          di_gemm_b1_tpp(
              grad_gelu[s1][nk], wt_TV[nc][nk], grad_in[s1][nc], Nkb);
      }
    }
  }
#else
  auto loop_scheme = large_cache_opt ? "acB" : "aBC";
  auto di_loop = ThreadedLoop<3>(
      {LoopSpecs{0, Nk, Nkb, false}, LoopSpecs{S1}, LoopSpecs{Nc}},
      loop_scheme);
  // auto di_loop =
  //    ThreadedLoop<3>({{0, Nk, Nkb, false}, {S1}, {Nc}}, loop_scheme);
  di_loop(
      [&](int* ind) {
        int nk = ind[0], s1 = ind[1], nc = ind[2];
        if (nk == 0)
          di_gemm_b0_tpp(
              grad_gelu[s1][nk], wt_TV[nc][nk], grad_in[s1][nc], Nkb, true);
        else
          di_gemm_b1_tpp(
              grad_gelu[s1][nk], wt_TV[nc][nk], grad_in[s1][nc], Nkb, true);
      },
      [&]() { di_gemm_b0_tpp.config(); },
      [&]() { di_gemm_b0_tpp.release(); });
#endif
}
{
  RECORD_SCOPE(dwi_gemm, {t_in_T, t_grad_gelu_V});
#ifdef NO_PARLOOPER
  for (int s1 = 0; s1 < S1; s1 += BS) {
    int count = (s1 + BS <= S1 ? BS : S1 - s1);
#pragma omp parallel for collapse(2)
    for (int nc = 0; nc < Nc; nc++) {
      for (int nk = 0; nk < Nk; nk++) {
        if (s1 == 0)
          dw_set_zero_tpp(grad_wt[nk][nc]);
        dw_gemm_tpp(
            in_T[in_blk(s1, nc)],
            grad_gelu_V[gdout_blk(s1, nk)],
            grad_wt[nk][nc],
            count);
        bool is_last_iter = !(s1 + BS < S1);
        if (grad_wt_flag != XformTPP::XFORM_NONE_TPP && is_last_iter) {
          T tmp[Hc * Hk];
          dw_cpy_tpp(grad_wt[nk][nc], tmp);
          dw_n2v_tpp(tmp, grad_wt[nk][nc]);
        }
      }
    }
  }
#else
  // auto dw_loop = ThreadedLoop<3>({{0, S1, BS, false}, {Nk}, {Nc}}, "aBC");
  auto dw_loop = ThreadedLoop<3>(
      {LoopSpecs{0, S1, BS, false}, LoopSpecs{Nk}, LoopSpecs{Nc}}, "aBC");
  dw_loop(
      [&](int* ind) {
        int s1 = ind[0], nk = ind[1], nc = ind[2];
        int count = (s1 + BS <= S1 ? BS : S1 - s1);
        if (s1 == 0)
          dw_set_zero_tpp(grad_wt[nk][nc]);
        dw_gemm_tpp(
            in_T[in_blk(s1, nc)],
            grad_gelu_V[gdout_blk(s1, nk)],
            grad_wt[nk][nc],
            count,
            true);
        bool is_last_iter = !(s1 + BS < S1);
        if (grad_wt_flag != XformTPP::XFORM_NONE_TPP && is_last_iter) {
          T tmp[Hc * Hk];
          dw_cpy_tpp(grad_wt[nk][nc], tmp);
          dw_n2v_tpp(tmp, grad_wt[nk][nc]);
        }
      },
      [&]() { dw_gemm_tpp.config(); },
      [&]() { dw_gemm_tpp.release(); });
#endif
}
return std::vector<at::Tensor>({t_grad_in, t_grad_wt, t_grad_bias});

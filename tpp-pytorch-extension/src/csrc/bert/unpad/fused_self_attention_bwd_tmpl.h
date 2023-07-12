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
int i = 0;
auto t_dCL = inputs[i++];
auto t_dAPO = inputs[i++];
auto t_Wq = inputs[i++]; // [HS][NH]
auto t_Wk = inputs[i++]; // [HS][NH]
auto t_Wv = inputs[i++]; // [HS][NH]
auto t_HS_T = inputs[i++]; // [B][S][HS]
auto t_HM = inputs[i++]; // Optional [B][N][S][S]
auto t_EHS_T = inputs[i++]; // [B][S][HS]
auto t_QL_T = inputs[i++];
auto t_KL_V = inputs[i++];
auto t_VL_TV = inputs[i++];
auto t_AP = inputs[i++];
auto t_APD_T = inputs[i++];
auto t_APD_mask = inputs[i++];
auto t_offs = inputs[i++]; // [B+1]
auto t_offs2 = inputs[i++]; // [B+1]

long B = t_offs.sizes()[0] - 1;
long SS1 = t_offs2[B].item().to<long>();
auto sizes = t_dCL.sizes();
auto S1 = sizes[0];
long N = sizes[1];
auto S2 = sizes[2];
long H = sizes[3];
// long NH = N*H;
float one_by_sqrt_H = 1.0 / sqrt(H);
bool dt_low_prec = (t_dCL.dtype() != at::kFloat);
const bool S2_eq_H = (S2 == H);
const long BS = 8;

auto t_dQL = t_QL_T.new_empty({S1, N, S2, H});
auto t_dQL_V = t_dQL;
auto t_dKL = t_KL_V.new_empty({S1, N, S2, H});
auto t_dKL_V = t_dKL;
auto t_dVL = t_VL_TV.new_empty({S1, N, S2, H});
auto t_dVL_V = t_dVL;

auto t_dWq = t_QL_T.new_empty({N, N, H, H});
auto t_dWk = t_QL_T.new_empty({N, N, H, H});
auto t_dWv = t_QL_T.new_empty({N, N, H, H});

auto t_dBq = t_QL_T.new_empty({N * H});
auto t_dBk = t_QL_T.new_empty({N * H});
auto t_dBv = t_QL_T.new_empty({N * H});

auto t_dHS = t_QL_T.new_empty({S1, N, S2, H});
at::Tensor t_dEHS;

auto t_dAPD = at::empty_like(t_AP);
auto t_dAPD_V = t_AP.new_empty({N, SS1, S2, S2});

auto null_EHS = false;
const int VBS = get_vnni_block_size<T>();

if (t_EHS_T.numel() == 0) {
  null_EHS = true;
  t_EHS_T = t_HS_T;
  t_dEHS = t_dHS;
} else {
  t_dEHS = t_QL_T.new_empty({S1, N, S2, H});
}

auto t_dCL_V = t_dCL;
if (dt_low_prec) {
  t_dQL_V = t_QL_T.new_empty({N, S1, S2 / VBS, H, VBS});
  t_dKL_V = t_KL_V.new_empty({N, S1, S2 / VBS, H, VBS});
  t_dVL_V = t_VL_TV.new_empty({N, S1, S2 / VBS, H, VBS});
  t_dCL_V = act_tensor_n2v_compact(S1, N, S2, H, t_dCL);
}
auto atrans_blk = LToPBlockAccessMapper<T>(S1, N);
const auto grad_wt_flag =
    (t_Wq.dim() == 5 ? XformTPP::XFORM_N2V_TPP : XformTPP::XFORM_NONE_TPP);
const auto a_trans_flag =
    (dt_low_prec ? XformTPP::XFORM_NONE_TPP : XformTPP::XFORM_XPOSE_TPP);
if (grad_wt_flag == XformTPP::XFORM_N2V_TPP) {
  t_dWq = t_dWq.view({N, N, H / VBS, H, VBS});
  t_dWk = t_dWk.view({N, N, H / VBS, H, VBS});
  t_dWv = t_dWv.view({N, N, H / VBS, H, VBS});
  t_dAPD_V = t_dAPD_V.view({N, SS1, S2 / VBS, S2, VBS});
}
auto t_Wq_TV = wt_tensor_for_bwd_compact(N, H, N, H, t_Wq);
auto t_Wk_TV = wt_tensor_for_bwd_compact(N, H, N, H, t_Wk);
auto t_Wv_TV = wt_tensor_for_bwd_compact(N, H, N, H, t_Wv);

{
  auto Wq_TV = GetVLAPtr<T>(t_Wq_TV, {N, H * H});
  auto Wk_TV = GetVLAPtr<T>(t_Wk_TV, {N, H * H});
  auto Wv_TV = GetVLAPtr<T>(t_Wv_TV, {N, H * H});
  auto dWq = GetVLAPtr<T>(t_dWq, {N, H * H});
  auto dWk = GetVLAPtr<T>(t_dWk, {N, H * H});
  auto dWv = GetVLAPtr<T>(t_dWv, {N, H * H});
  auto dBq = GetVLAPtr<T>(t_dBq, {H});
  auto dBk = GetVLAPtr<T>(t_dBk, {H});
  auto dBv = GetVLAPtr<T>(t_dBv, {H});
  auto QL_T = GetVLAPtr<T>(t_QL_T, {H * S2});
  auto KL_V = GetVLAPtr<T>(t_KL_V, {N, S2 * H});
  auto VL_TV = GetVLAPtr<T>(t_VL_TV, {N, H * S2});
  auto dQL = GetVLAPtr<T>(t_dQL, {N, S2 * H});
  auto dQL_V = GetVLAPtr<T>(t_dQL_V, {S2 * H});
  auto dKL = GetVLAPtr<T>(t_dKL, {N, S2 * H});
  auto dKL_V = GetVLAPtr<T>(t_dKL_V, {S2 * H});
  auto dVL = GetVLAPtr<T>(t_dVL, {N, S2 * H});
  auto dVL_V = GetVLAPtr<T>(t_dVL_V, {S2 * H});
  auto AP = GetVLAPtr<T>(t_AP, {SS1, S2 * S2});
  auto APD_mask = GetVLAPtr<short>(t_APD_mask, {SS1, (S2 * S2 + 15) / 16});
  auto dCL = GetVLAPtr<T>(t_dCL, {N, S2 * H});
  auto dCL_V = GetVLAPtr<T>(t_dCL_V, {S2 * H});
  auto APD_T = GetVLAPtr<T>(t_APD_T, {SS1, S2 * S2});
  auto dAPO = GetVLAPtr<T>(t_dAPO, {SS1, S2 * S2});
  auto dAPD_V = GetVLAPtr<T>(t_dAPD_V, {SS1, S2 * S2});
  auto HS_T = GetVLAPtr<T>(t_HS_T, {H * S2});
  auto EHS_T = GetVLAPtr<T>(t_EHS_T, {H * S2});
  auto dHS = GetVLAPtr<T>(t_dHS, {N, S2 * H});
  auto dEHS = GetVLAPtr<T>(t_dEHS, {N, S2 * H});
  auto offs = t_offs.data_ptr<long>();
  auto offs2 = t_offs2.data_ptr<long>();

  auto cw_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2,
      H,
      S2,
      S2 * S2,
      dt_low_prec ? S2 * H : N * S2 * H,
      0.0,
      XformTPP::XFORM_NONE_TPP,
      0 /*a_trans_flag*/, // We transpose in FWD to have fixed stride of blocks
      1)));
  auto cw_n2v_tpp =
      SCOPEIT(XformExtTPP<T>(S2, H, XformTPP::XFORM_N2V_TPP, true), VNNI);
  auto a_convert_tpp = SCOPEIT((ConvertTPP<T, float>(S2, S2)), EW_COPY);
  auto ci_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, float>(
      S2,
      S2,
      H,
      S2 * H,
      S2 * H,
      dAPO ? 1.0 : 0.0,
      XformTPP::XFORM_NONE_TPP,
      0,
      1)));
  auto dropout_bwd_tpp = SCOPEIT(DropOutBwdTPP<float>(S2 * S2, p), DROPOUT);
  auto softmax_bwd_tpp =
      SCOPEIT((VarSoftMaxBwdTPP<float, float, T>(S2, S2)), SOFTMAX);
  auto scale_tpp = SCOPEIT((ScaleTPP<float, T>(S2 * S2)), EW_SCL);
  auto a_n2v_tpp =
      SCOPEIT(XformExtTPP<T>(S2, S2, XformTPP::XFORM_N2V_TPP, true), VNNI);
  auto ai_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2, H, S2, S2 * S2, N * S2 * H, 0.0, XformTPP::XFORM_NONE_TPP, 0, S1)));
  auto aw_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      H,
      S2,
      S2,
      a_trans_flag == XformTPP::XFORM_NONE_TPP ? S2 * H : N * S2 * H,
      S2 * S2,
      0.0,
      XformTPP::XFORM_XPOSE_TPP,
      a_trans_flag,
      1)));
  auto vi_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2, H, H, S2 * H, H * H, 0.0, XformTPP::XFORM_NONE_TPP, 0, N)));
  auto ki_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2, H, H, S2 * H, H * H, 1.0, XformTPP::XFORM_NONE_TPP, 0, N)));
  auto qi_gemm_tpp = (null_EHS ? ki_gemm_tpp : vi_gemm_tpp);
  auto dw_set_zero_tpp = SCOPEIT(SetZeroTPP<T>(H * H), EW_ZERO);
  auto dw_cpy_tpp = SCOPEIT(CpyTPP<T>(H * H), VNNI);
  auto dw_n2v_tpp =
      SCOPEIT(XformExtTPP<T>(H, H, XformTPP::XFORM_N2V_TPP, true), VNNI);
  auto qkvw_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      H,
      H,
      S2,
      a_trans_flag == XformTPP::XFORM_NONE_TPP ? S2 * H : N * S2 * H,
      a_trans_flag == XformTPP::XFORM_NONE_TPP ? S2 * H : N * S2 * H,
      1.0,
      XformTPP::XFORM_NONE_TPP, //(XformTPP::XFORM_TYPE)grad_wt_flag,
      a_trans_flag,
      BS)));
  auto set_zero_dw_tpp = SCOPEIT(SetZeroTPP<T>(H * H), EW_ZERO);
  auto set_zero_f32_tpp = SCOPEIT(SetZeroTPP<float>(N * H), EW_ZERO);
  auto grad_bias_tpp = SCOPEIT(GradBiasTPP<T>(S2, H), BIAS);

  {
    RECORD_SCOPE(dac_gemm, {t_APD_T, t_dCL_V});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
      // dVL = APD_T * dCL
#pragma omp parallel for collapse(2) schedule(static, 1)
      for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
          long start = offs[b];
          // long ss1 = offs2[b];
          long end = offs[b + 1];
          long len = end - start;
          cw_gemm_tpp.config();
          for (int s21 = start, ss1 = offs2[b]; s21 < end; s21++, ss1 += len) {
            cw_gemm_tpp(
                APD_T[n][ss1],
                dCL_V[atrans_blk(start, n)],
                dVL[s21][n],
                len,
                true);
            if (dt_low_prec)
              cw_n2v_tpp(dVL[s21][n], dVL_V[atrans_blk(s21, n)]);
          }
          if (!S2_eq_H)
            cw_gemm_tpp.release();
          for (int s11 = start, ss1 = offs2[b]; s11 < end; s11++, ss1 += len) {
            float dtAPD[len][S2][S2];
            T dtAPD_bf[len][S2][S2];
            if (!S2_eq_H)
              ci_gemm_tpp.config();
            for (int s21 = start; s21 < end; s21++) {
              auto ls21 = s21 - start;
              if (dAPO)
                a_convert_tpp(dAPO[n][ss1 + ls21], dtAPD[ls21][0]);
              ci_gemm_tpp(dCL[s11][n], VL_TV[s21][n], dtAPD[ls21][0], 1, true);
            }
            if (!S2_eq_H)
              ci_gemm_tpp.release();
            if (t_HM.numel() != 0) {
              // FIXME: shape of head mask is not correct here yet
              TPP_ASSERT(0, "t_HM used");
              // t_dAPD[b][s11][n] = t_dAPD[b][s11][n] * t_HM[b][s11][n];
            }
            if (p > 0) {
              for (int l = 0; l < len; l++) {
                dropout_bwd_tpp(dtAPD[l][0], dtAPD[l][0], APD_mask[n][ss1 + l]);
              }
            }
            softmax_bwd_tpp(len, dtAPD[0][0], dtAPD[0][0], AP[n][ss1]);
            for (int s21 = start; s21 < end; s21++) {
              auto ls21 = s21 - start;
              long l = s11 - start;
              long ss = offs2[b];
              scale_tpp(dtAPD[ls21][0], dtAPD_bf[ls21][0], one_by_sqrt_H);
              a_n2v_tpp(dtAPD_bf[ls21][0], dAPD_V[n][ss + ls21 * len + l]);
            }
            // dQL = dADP * KL_V
            ai_gemm_tpp(
                dtAPD_bf[0][0], KL_V[start][n], dQL[s11][n], len, S2_eq_H);
            if (dt_low_prec)
              cw_n2v_tpp(dQL[s11][n], dQL_V[atrans_blk(s11, n)]);
          }
          if (!S2_eq_H)
            aw_gemm_tpp.config();
          for (int s21 = start, ss1 = offs2[b]; s21 < end; s21++, ss1 += len) {
            // dKL = (QL_T * dAPD)T
            aw_gemm_tpp(
                QL_T[atrans_blk(start, n)],
                dAPD_V[n][ss1],
                dKL[s21][n],
                len,
                true);
            if (dt_low_prec)
              cw_n2v_tpp(dKL[s21][n], dKL_V[atrans_blk(s21, n)]);
          }
          // The if condition below is just to match config / release on same
          // tpp
          if (!S2_eq_H) {
            aw_gemm_tpp.release();
          } else {
            cw_gemm_tpp.release();
          }
        }
      }
    }
  }

#ifndef NO_PARLOOPER
  auto loop_scheme = large_cache_opt ? "bA" : "AB";
  // auto qkv_loop = ThreadedLoop<2>({{S1}, {N}}, loop_scheme);
  auto qkv_loop = ThreadedLoop<2>({LoopSpecs{S1}, LoopSpecs{N}}, loop_scheme);
#endif
  {
    RECORD_SCOPE(div_gemm, {t_dVL, t_Wv_TV});
    {
#ifdef NO_PARLOOPER
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int nc = 0; nc < N; nc++) {
          vi_gemm_tpp(dVL[s1][0], Wv_TV[nc][0], dEHS[s1][nc], N);
        }
      }
#else
      qkv_loop(
          [&](int* ind) {
            int s1 = ind[0], nc = ind[1];
            vi_gemm_tpp(dVL[s1][0], Wv_TV[nc][0], dEHS[s1][nc], N, true);
          },
          [&]() { vi_gemm_tpp.config(); },
          [&]() { vi_gemm_tpp.release(); });
#endif
    }
  }
  {
    RECORD_SCOPE(dik_gemm, {t_dKL, t_Wk_TV});
    {
#ifdef NO_PARLOOPER
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int nc = 0; nc < N; nc++) {
          ki_gemm_tpp(dKL[s1][0], Wk_TV[nc][0], dEHS[s1][nc], N);
        }
      }
#else
      qkv_loop(
          [&](int* ind) {
            int s1 = ind[0], nc = ind[1];
            ki_gemm_tpp(dKL[s1][0], Wk_TV[nc][0], dEHS[s1][nc], N, true);
          },
          [&]() { ki_gemm_tpp.config(); },
          [&]() { ki_gemm_tpp.release(); });
#endif
    }
  }
  {
    RECORD_SCOPE(diq_gemm, {t_dQL, t_Wq_TV});
    {
#ifdef NO_PARLOOPER
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int nc = 0; nc < N; nc++) {
          qi_gemm_tpp(dQL[s1][0], Wq_TV[nc][0], dHS[s1][nc], N);
        }
      }
#else
      qkv_loop(
          [&](int* ind) {
            int s1 = ind[0], nc = ind[1];
            qi_gemm_tpp(dQL[s1][0], Wq_TV[nc][0], dHS[s1][nc], N, true);
          },
          [&]() { qi_gemm_tpp.config(); },
          [&]() { qi_gemm_tpp.release(); });
#endif
    }
  }
  {
    RECORD_SCOPE(dwqkv_gemm, {t_HS_T, t_dQL_V});
#ifdef NO_PARLOOPER
    for (int s1 = 0; s1 < S1; s1 += BS) {
      int count = (s1 + BS <= S1 ? BS : S1 - s1);
      bool is_last_iter = !(s1 + BS < S1);
#pragma omp parallel for collapse(2)
      for (int nk = 0; nk < N; nk++) {
        for (int nc = 0; nc < N; nc++) {
          if (s1 == 0) {
            set_zero_dw_tpp(dWv[nk][nc]);
            set_zero_dw_tpp(dWk[nk][nc]);
            set_zero_dw_tpp(dWq[nk][nc]);
          }
          qkvw_gemm_tpp(
              EHS_T[atrans_blk(s1, nc)],
              dVL_V[atrans_blk(s1, nk)],
              dWv[nk][nc],
              count);
          if (grad_wt_flag != XformTPP::XFORM_NONE_TPP && is_last_iter) {
            T tmp[H * H];
            dw_cpy_tpp(dWv[nk][nc], tmp);
            dw_n2v_tpp(tmp, dWv[nk][nc]);
          }
          qkvw_gemm_tpp(
              EHS_T[atrans_blk(s1, nc)],
              dKL_V[atrans_blk(s1, nk)],
              dWk[nk][nc],
              count);
          if (grad_wt_flag != XformTPP::XFORM_NONE_TPP && is_last_iter) {
            T tmp[H * H];
            dw_cpy_tpp(dWk[nk][nc], tmp);
            dw_n2v_tpp(tmp, dWk[nk][nc]);
          }
          qkvw_gemm_tpp(
              HS_T[atrans_blk(s1, nc)],
              dQL_V[atrans_blk(s1, nk)],
              dWq[nk][nc],
              count);
          if (grad_wt_flag != XformTPP::XFORM_NONE_TPP && is_last_iter) {
            T tmp[H * H];
            dw_cpy_tpp(dWq[nk][nc], tmp);
            dw_n2v_tpp(tmp, dWq[nk][nc]);
          }
        }
      }
    }
#else
    // auto qkvw_loop = ThreadedLoop<3>({{0, S1, BS, false}, {N}, {N}}, "aBC");
    auto qkvw_loop = ThreadedLoop<3>(
        {LoopSpecs{0, S1, BS, false}, LoopSpecs{N}, LoopSpecs{N}}, "aBC");
    qkvw_loop(
        [&](int* ind) {
          int s1 = ind[0], nk = ind[1], nc = ind[2];
          int count = (s1 + BS <= S1 ? BS : S1 - s1);
          bool is_last_iter = !(s1 + BS < S1);
          if (s1 == 0) {
            set_zero_dw_tpp(dWv[nk][nc]);
            set_zero_dw_tpp(dWk[nk][nc]);
            set_zero_dw_tpp(dWq[nk][nc]);
          }
          qkvw_gemm_tpp(
              EHS_T[atrans_blk(s1, nc)],
              dVL_V[atrans_blk(s1, nk)],
              dWv[nk][nc],
              count,
              true);
          if (grad_wt_flag != XformTPP::XFORM_NONE_TPP && is_last_iter) {
            T tmp[H * H];
            dw_cpy_tpp(dWv[nk][nc], tmp);
            dw_n2v_tpp(tmp, dWv[nk][nc]);
          }
          qkvw_gemm_tpp(
              EHS_T[atrans_blk(s1, nc)],
              dKL_V[atrans_blk(s1, nk)],
              dWk[nk][nc],
              count,
              true);
          if (grad_wt_flag != XformTPP::XFORM_NONE_TPP && is_last_iter) {
            T tmp[H * H];
            dw_cpy_tpp(dWk[nk][nc], tmp);
            dw_n2v_tpp(tmp, dWk[nk][nc]);
          }
          qkvw_gemm_tpp(
              HS_T[atrans_blk(s1, nc)],
              dQL_V[atrans_blk(s1, nk)],
              dWq[nk][nc],
              count,
              true);
          if (grad_wt_flag != XformTPP::XFORM_NONE_TPP && is_last_iter) {
            T tmp[H * H];
            dw_cpy_tpp(dWq[nk][nc], tmp);
            dw_n2v_tpp(tmp, dWq[nk][nc]);
          }
        },
        [&]() { qkvw_gemm_tpp.config(); },
        [&]() { qkvw_gemm_tpp.release(); });
#endif
  }
  {
    RECORD_SCOPE(dqkv_bias, {t_dQL});
    int num_threads = omp_get_max_threads();
    float* bias_ptrs[num_threads];
    {
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        float prv_grad_bias[N][H];
        bias_ptrs[tid] = prv_grad_bias[0];
        set_zero_f32_tpp(prv_grad_bias[0]);
#pragma omp for collapse(2)
        for (int s1 = 0; s1 < S1; s1++) {
          for (int n = 0; n < N; n++) {
            grad_bias_tpp(dQL[s1][n], prv_grad_bias[n]);
          }
        }
        omp_reduce_buf(num_threads, N * H, bias_ptrs, dBq[0]);
        set_zero_f32_tpp(prv_grad_bias[0]);
#pragma omp for collapse(2)
        for (int s1 = 0; s1 < S1; s1++) {
          for (int n = 0; n < N; n++) {
            grad_bias_tpp(dKL[s1][n], prv_grad_bias[n]);
          }
        }
        omp_reduce_buf(num_threads, N * H, bias_ptrs, dBk[0]);
        set_zero_f32_tpp(prv_grad_bias[0]);
#pragma omp for collapse(2)
        for (int s1 = 0; s1 < S1; s1++) {
          for (int n = 0; n < N; n++) {
            grad_bias_tpp(dVL[s1][n], prv_grad_bias[n]);
          }
        }
        omp_reduce_buf(num_threads, N * H, bias_ptrs, dBv[0]);
      }
    }
  }
  if (null_EHS) {
    t_dEHS = at::Tensor();
  }
}
return std::vector<at::Tensor>(
    {t_dWq, t_dBq, t_dWk, t_dBk, t_dWv, t_dBv, t_dHS, t_dEHS});

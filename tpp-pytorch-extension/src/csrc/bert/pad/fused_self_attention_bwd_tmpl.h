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

auto sizes = t_dCL.sizes();
auto B = sizes[0];
auto S1 = sizes[1];
long N = sizes[2];
auto S2 = sizes[3];
long H = sizes[4];
// long NH = N*H;
float one_by_sqrt_H = 1.0 / sqrt(H);
bool dt_bf16 = (t_dCL.dtype() == at::kBFloat16);

auto t_dQL = t_QL_T.new_empty({B, S1, N, S2, H});
auto t_dQL_V = t_dQL;
auto t_dKL = t_KL_V.new_empty({B, S1, N, S2, H});
auto t_dKL_V = t_dKL;
auto t_dVL = t_VL_TV.new_empty({B, S1, N, S2, H});
auto t_dVL_V = t_dVL;

auto t_dWq = t_QL_T.new_empty({N, N, H, H});
auto t_dWk = t_QL_T.new_empty({N, N, H, H});
auto t_dWv = t_QL_T.new_empty({N, N, H, H});

auto t_dBq = t_QL_T.new_empty({N * H});
auto t_dBk = t_QL_T.new_empty({N * H});
auto t_dBv = t_QL_T.new_empty({N * H});

auto t_dHS = t_QL_T.new_empty({B, S1, N, S2, H});
// auto t_dEHS = t_QL.new_empty({B, S1, N, S2, H});
at::Tensor t_dEHS; // = t_QL.new_empty({B, S1, N, S2, H});

auto t_dAPD = at::empty_like(t_AP);
// auto t_dAPD_V = at::empty_like(t_dAPO);
auto t_dAPD_V = t_AP.new_empty({B, S1, N, S1, S2, S2});

auto null_EHS = false;

if (t_EHS_T.numel() == 0) {
  null_EHS = true;
  t_EHS_T = t_HS_T;
  t_dEHS = t_dHS;
} else {
  t_dEHS = t_QL_T.new_empty({B, S1, N, S2, H});
}

auto t_dCL_V = t_dCL;
if (dt_bf16) {
  t_dQL_V = t_QL_T.new_empty({B, S1, N, S2 / 2, H, 2});
  t_dKL_V = t_KL_V.new_empty({B, S1, N, S2 / 2, H, 2});
  t_dVL_V = t_VL_TV.new_empty({B, S1, N, S2 / 2, H, 2});
  t_dCL_V = act_tensor_n2v(B, S1, N, S2, H, t_dCL);
}
const auto grad_wt_flag =
    (t_Wq.dim() == 5 ? XformTPP::XFORM_N2V_TPP : XformTPP::XFORM_NONE_TPP);
const auto a_trans_flag =
    (dt_bf16 ? XformTPP::XFORM_NONE_TPP : XformTPP::XFORM_XPOSE_TPP);
if (grad_wt_flag == XformTPP::XFORM_N2V_TPP) {
  t_dWq = t_dWq.view({N, N, H / 2, H, 2});
  t_dWk = t_dWk.view({N, N, H / 2, H, 2});
  t_dWv = t_dWv.view({N, N, H / 2, H, 2});
  t_dAPD_V = t_dAPD_V.view({B, S1, N, S1, S2 / 2, S2, 2});
}
auto t_Wq_TV = wt_tensor_for_bwd(N, H, N, H, t_Wq);
auto t_Wk_TV = wt_tensor_for_bwd(N, H, N, H, t_Wk);
auto t_Wv_TV = wt_tensor_for_bwd(N, H, N, H, t_Wv);

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
  auto QL_T = GetVLAPtr<T>(t_QL_T, {S1, N, H * S2});
  auto KL_V = GetVLAPtr<T>(t_KL_V, {S1, N, S2 * H});
  auto VL_TV = GetVLAPtr<T>(t_VL_TV, {S1, N, H * S2});
  auto dQL = GetVLAPtr<T>(t_dQL, {S1, N, S2 * H});
  auto dQL_V = GetVLAPtr<T>(t_dQL_V, {S1, N, S2 * H});
  auto dKL = GetVLAPtr<T>(t_dKL, {S1, N, S2 * H});
  auto dKL_V = GetVLAPtr<T>(t_dKL_V, {S1, N, S2 * H});
  auto dVL = GetVLAPtr<T>(t_dVL, {S1, N, S2 * H});
  auto dVL_V = GetVLAPtr<T>(t_dVL_V, {S1, N, S2 * H});
  auto AP = GetVLAPtr<T>(t_AP, {S1, N, S1, S2 * S2});
  auto APD_mask =
      GetVLAPtr<short>(t_APD_mask, {S1, N, (S1 * S2 * S2 + 15) / 16});
  auto dCL = GetVLAPtr<T>(t_dCL, {S1, N, S2 * H});
  auto dCL_V = GetVLAPtr<T>(t_dCL_V, {S1, N, S2 * H});
  auto APD_T = GetVLAPtr<T>(t_APD_T, {S1, N, S1, S2 * S2});
  auto dAPO = GetVLAPtr<T>(t_dAPO, {S1, N, S1, S2 * S2});
  auto dAPD_V = GetVLAPtr<T>(t_dAPD_V, {S1, N, S1, S2 * S2});
  auto HS_T = GetVLAPtr<T>(t_HS_T, {S1, N, H * S2});
  auto EHS_T = GetVLAPtr<T>(t_EHS_T, {S1, N, H * S2});
  auto dHS = GetVLAPtr<T>(t_dHS, {S1, N, S2 * H});
  auto dEHS = GetVLAPtr<T>(t_dEHS, {S1, N, S2 * H});

  auto cw_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2,
      H,
      S2,
      N * S1 * S2 * S2,
      N * S2 * H,
      0.0,
      XformTPP::XFORM_NONE_TPP,
      a_trans_flag,
      S1)));
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
  auto dropout_bwd_tpp =
      SCOPEIT(DropOutBwdTPP<float>(S1 * S2 * S2, p), DROPOUT);
  auto softmax_bwd_tpp =
      SCOPEIT((SoftMaxBwdTPP<float, float, T>(S1, S2, S2)), SOFTMAX);
  auto scale_tpp = SCOPEIT((ScaleTPP<float, T>(S2 * S2)), EW_SCL);
  auto a_n2v_tpp =
      SCOPEIT(XformExtTPP<T>(S2, S2, XformTPP::XFORM_N2V_TPP, true), VNNI);
  auto ai_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2, H, S2, S2 * S2, N * S2 * H, 0.0, XformTPP::XFORM_NONE_TPP, 0, S1)));
  auto aw_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      H,
      S2,
      S2,
      N * S2 * H,
      N * S1 * S2 * S2,
      0.0,
      XformTPP::XFORM_XPOSE_TPP,
      a_trans_flag,
      S1)));
  auto vi_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2, H, H, S2 * H, N * H * H, 0.0, XformTPP::XFORM_NONE_TPP, 0, N)));
  auto ki_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2, H, H, S2 * H, N * H * H, 1.0, XformTPP::XFORM_NONE_TPP, 0, N)));
  auto qi_gemm_tpp = (null_EHS ? ki_gemm_tpp : vi_gemm_tpp);
  auto qkvw_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      H,
      H,
      S2,
      N * S2 * H,
      N * S2 * H,
      1.0,
      (XformTPP::XFORM_TYPE)(grad_wt_flag),
      a_trans_flag,
      S1)));
  // auto set_zero_dt_tpp = SCOPEIT(SetZeroTPP<T>(N*H), EW_ZERO);
  auto set_zero_f32_tpp = SCOPEIT(SetZeroTPP<float>(N * H), EW_ZERO);
  auto grad_bias_tpp = SCOPEIT(GradBiasTPP<T>(S2, H), BIAS);

  // printf("dAPO = %p, t_dAPO.size = %lu\n", dAPO, t_dAPO.numel());
  // #define PRINT_T(x) std::cout << #x << ": " << x << std::endl
  // #define PRINT_T(x)
  {
    RECORD_SCOPE(dwc_gemm, {t_APD_T, t_dCL_V});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
      // dVL = APD_T * dCL
#pragma omp parallel for collapse(3)
      for (int b = 0; b < B; b++) {
        for (int s21 = 0; s21 < S1; s21++) {
          for (int n = 0; n < N; n++) {
            cw_gemm_tpp(
                APD_T[b][0][n][s21], dCL_V[b][0][n], dVL[b][s21][n], S1);
            if (dt_bf16)
              cw_n2v_tpp(dVL[b][s21][n], dVL_V[b][s21][n]);
          }
        }
      }
    }
  }
  // PRINT_T(t_AP);
  {
    RECORD_SCOPE(dica_gemm, {t_dCL, t_VL_TV});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
      // dAPD = dCL * VL_TV
#pragma omp parallel for collapse(3)
      for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
          for (int s11 = 0; s11 < S1; s11++) {
            float dtAPD[S1][S2][S2] = {0};
            T dtAPD_bf[S1][S2][S2] = {0};
            for (int s21 = 0; s21 < S1; s21++) {
              if (dAPO)
                a_convert_tpp(dAPO[b][s11][n][s21], dtAPD[s21][0]);
              ci_gemm_tpp(dCL[b][s11][n], VL_TV[b][s21][n], dtAPD[s21][0], 1);
            }
            if (t_HM.numel() != 0) {
              // FIXME: shape of head mask is not correct here yet
              TPP_ASSERT(0, "t_HM used");
              // t_dAPD[b][s11][n] = t_dAPD[b][s11][n] * t_HM[b][s11][n];
            }
            if (p > 0) {
              dropout_bwd_tpp(dtAPD[0][0], dtAPD[0][0], APD_mask[b][s11][n]);
            }
            softmax_bwd_tpp(dtAPD[0][0], dtAPD[0][0], AP[b][s11][n][0]);
            for (int s21 = 0; s21 < S1; s21++) {
              scale_tpp(dtAPD[s21][0], dtAPD_bf[s21][0], one_by_sqrt_H);
              a_n2v_tpp(dtAPD_bf[s21][0], dAPD_V[b][s11][n][s21]);
            }
            // dQL = dADP * KL_V
            ai_gemm_tpp(dtAPD_bf[0][0], KL_V[b][0][n], dQL[b][s11][n], S1);
            if (dt_bf16)
              cw_n2v_tpp(dQL[b][s11][n], dQL_V[b][s11][n]);
          }
        }
      }
    }
  }
  {
    RECORD_SCOPE(dwa_gemm, {t_QL_T, t_dAPD_V});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
      for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
          for (int s21 = 0; s21 < S1; s21++) {
            // dKL = (QL_T * dAPD)T
            aw_gemm_tpp(
                QL_T[b][0][n], dAPD_V[b][0][n][s21], dKL[b][s21][n], S1);
            if (dt_bf16)
              cw_n2v_tpp(dKL[b][s21][n], dKL_V[b][s21][n]);
          }
        }
      }
    }
  }
  // PRINT_T(t_QL_T.permute({0,1,2,4,3}).contiguous());
  // PRINT_T(t_dAPD_V.permute({0,1,2,3,4,6,5}).contiguous().view({B,S1,N,S1,S2,S2}));
  // PRINT_T(t_dKL);
  {
    RECORD_SCOPE(div_gemm, {t_dVL, t_Wv_TV});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
      for (int b = 0; b < B; b++) {
        for (int s1 = 0; s1 < S1; s1++) {
          for (int nc = 0; nc < N; nc++) {
            vi_gemm_tpp(dVL[b][s1][0], Wv_TV[0][nc], dEHS[b][s1][nc], N);
          }
        }
      }
    }
  }
  {
    RECORD_SCOPE(dik_gemm, {t_dKL, t_Wk_TV});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
      for (int b = 0; b < B; b++) {
        for (int s1 = 0; s1 < S1; s1++) {
          for (int nc = 0; nc < N; nc++) {
            ki_gemm_tpp(dKL[b][s1][0], Wk_TV[0][nc], dEHS[b][s1][nc], N);
          }
        }
      }
    }
  }
  {
    RECORD_SCOPE(diq_gemm, {t_dQL, t_Wq_TV});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
      for (int b = 0; b < B; b++) {
        for (int s1 = 0; s1 < S1; s1++) {
          for (int nc = 0; nc < N; nc++) {
            qi_gemm_tpp(dQL[b][s1][0], Wq_TV[0][nc], dHS[b][s1][nc], N);
          }
        }
      }
    }
  }
  {
    RECORD_SCOPE(dwqkv_gemm, {t_HS_T, t_dQL_V});
#if 0
      t_dWv.zero_();
      t_dWk.zero_();
      t_dWq.zero_();
#else
    tensor_set_zero(N * N, H * H, t_dWv);
    tensor_set_zero(N * N, H * H, t_dWk);
    tensor_set_zero(N * N, H * H, t_dWq);
#endif
    for (int b = 0; b < B; b++) {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int nk = 0; nk < N; nk++) {
        for (int nc = 0; nc < N; nc++) {
          qkvw_gemm_tpp(EHS_T[b][0][nc], dVL_V[b][0][nk], dWv[nk][nc], S1);
          qkvw_gemm_tpp(EHS_T[b][0][nc], dKL_V[b][0][nk], dWk[nk][nc], S1);
          qkvw_gemm_tpp(HS_T[b][0][nc], dQL_V[b][0][nk], dWq[nk][nc], S1);
        }
      }
    }
  }
  // PRINT_T(t_EHS_T.permute({0,1,2,4,3}).contiguous());
  // PRINT_T(t_HS_T.permute({0,1,2,4,3}).contiguous());
  {
    RECORD_SCOPE(dqkv_bias, {t_dQL});
    int num_threads = omp_get_max_threads();
    float* bias_ptrs[num_threads];
#if 0
      t_dBq.zero_();
      t_dBk.zero_();
      t_dBv.zero_();
#else
    tensor_set_zero(N, H, t_dBq);
    tensor_set_zero(N, H, t_dBk);
    tensor_set_zero(N, H, t_dBv);
#endif
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        float prv_grad_bias[N][H];
        bias_ptrs[tid] = prv_grad_bias[0];
        set_zero_f32_tpp(prv_grad_bias[0]);
#pragma omp for collapse(3)
        for (int b = 0; b < B; b++) {
          for (int s1 = 0; s1 < S1; s1++) {
            for (int n = 0; n < N; n++) {
              grad_bias_tpp(dQL[b][s1][n], prv_grad_bias[n]);
            }
          }
        }
#pragma omp barrier
        omp_reduce_buf(num_threads, N * H, bias_ptrs, dBq[0]);
#pragma omp barrier
        set_zero_f32_tpp(prv_grad_bias[0]);
#pragma omp for collapse(3)
        for (int b = 0; b < B; b++) {
          for (int s1 = 0; s1 < S1; s1++) {
            for (int n = 0; n < N; n++) {
              grad_bias_tpp(dKL[b][s1][n], prv_grad_bias[n]);
            }
          }
        }
#pragma omp barrier
        omp_reduce_buf(num_threads, N * H, bias_ptrs, dBk[0]);
#pragma omp barrier
        set_zero_f32_tpp(prv_grad_bias[0]);
#pragma omp for collapse(3)
        for (int b = 0; b < B; b++) {
          for (int s1 = 0; s1 < S1; s1++) {
            for (int n = 0; n < N; n++) {
              grad_bias_tpp(dVL[b][s1][n], prv_grad_bias[n]);
            }
          }
        }
#pragma omp barrier
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

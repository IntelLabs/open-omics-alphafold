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
// B - Batch size
// S - Max seq len
// N - Number of attention heads
// H - Head size
auto t_Wq = inputs[0]; // [HS][NH] --> [N1][N2][H2][H1]
auto t_Bq = inputs[1]; // [HS]
auto t_Wk = inputs[2]; // [HS][NH] --> [N1][N2][H2][H1]
auto t_Bk = inputs[3]; // [HS]
auto t_Wv = inputs[4]; // [HS][NH] --> [N1][N2][H2][H1]
auto t_Bv = inputs[5]; // [HS]
auto t_HS = inputs[6]; // [B][S][HS] --> [B][S1][N][S2][H]
auto t_AM = inputs[7]; // Optional [B][S]
auto t_HM = inputs[8]; // Optional [B][N][S][S]
auto t_EHS = inputs[9]; // [B][S][HS] --> [B][S1][N][S2][H]
auto t_EAM = inputs[10]; // Optional [B][S]
auto t_offs = inputs[11]; // [B+1]
auto t_offs2 = inputs[12]; // [B+1]

long B = t_offs.sizes()[0] - 1;
long SS1 = t_offs2[B].item().to<long>();
auto sizes = t_HS.sizes();
long S1 = sizes[0];
long N = sizes[1];
long S2 = sizes[2];
long H = sizes[3];
// long NH = N*H;
float one_by_sqrt_H = 1.0 / sqrt(H);
bool null_EHS = false;
bool dt_low_prec = (t_HS.dtype() != at::kFloat);
bool low_prec_training = (training && dt_low_prec);
auto t_EHS_orig = t_EHS;

const int VBS = get_vnni_block_size<T>();
// std::cout << "B: " << B << " S1: " << S1 << " S2: " << S2 << " N: " << N << "
// H: " << H << std::endl;
if (t_EHS.numel() == 0) {
  null_EHS = true;
  t_EHS = t_HS;
} else {
  t_AM = t_EAM;
}

// #define PRINT_T(x) std::cout << #x << ": " << x << std::endl
auto t_HS_T = t_HS;
auto t_EHS_T = t_EHS;

auto t_Wq_V = wt_tensor_for_fwd(N, H, N, H, t_Wq);
auto t_Wk_V = wt_tensor_for_fwd(N, H, N, H, t_Wk);
auto t_Wv_V = wt_tensor_for_fwd(N, H, N, H, t_Wv);

auto t_QL = t_HS.new_empty({S1, N, S2, H});
auto t_QL_T = t_QL;
auto t_KL_TV = t_EHS.new_empty({S1, N, H, S2});
if (dt_low_prec)
  t_KL_TV = t_KL_TV.view({S1, N, H / VBS, S2, VBS});
auto t_KL_V = t_KL_TV;
auto t_VL_V = t_EHS.new_empty({S1, N, S2, H});
if (dt_low_prec)
  t_VL_V = t_VL_V.view({S1, N, S2 / VBS, H, VBS});
auto t_VL_TV = t_VL_V;
auto t_AP = t_QL.new_empty({N, SS1, S2, S2});
auto t_CL = t_AP.new_empty({S1, N, S2, H});

auto t_APD = t_AP;
auto t_APD_mask = at::empty({N, SS1, (S2 * S2 + 15) / 16}, at::kShort);
if (p > 0 || t_HM.numel() != 0) {
  t_APD = at::empty_like(t_AP);
}

auto t_APD_T = t_APD;

if (low_prec_training) {
  t_HS_T = t_HS.new_empty({N, S1, H, S2}); // For BWD only
  t_EHS_T = null_EHS ? t_HS_T : t_HS.new_empty({N, S1, H, S2}); // For BWD only

  t_QL_T = t_HS.new_empty({N, S1, H, S2}); // For BWD only
}
if (training) {
  if (dt_low_prec) {
    t_KL_V = t_EHS.new_empty({S1, N, S2 / VBS, H, VBS}); // Saved For BWD
    t_VL_TV = t_EHS.new_empty({S1, N, H / VBS, S2, VBS}); // For BWD only
  } else {
    t_KL_V = t_EHS.new_empty({S1, N, S2, H}); // Saved For BWD
    t_VL_TV = t_EHS.new_empty({S1, N, H, S2}); // For BWD only
  }
  t_APD_T = t_QL.new_empty({N, SS1, S2, S2}); // For BWD only
}

{
  auto Wq_V = GetVLAPtr<T>(t_Wq_V, {N, H * H});
  auto Wk_V = GetVLAPtr<T>(t_Wk_V, {N, H * H});
  auto Wv_V = GetVLAPtr<T>(t_Wv_V, {N, H * H});
  auto Bq = GetVLAPtr<T>(t_Bq, {H});
  auto Bk = GetVLAPtr<T>(t_Bk, {H});
  auto Bv = GetVLAPtr<T>(t_Bv, {H});
  auto QL = GetVLAPtr<T>(t_QL, {N, S2 * H});
  auto QL_T = GetVLAPtr<T>(t_QL_T, {S1, H * S2}); // For BWD only
  auto KL_V = GetVLAPtr<T>(t_KL_V, {N, S2 * H});
  auto KL_TV = GetVLAPtr<T>(t_KL_TV, {N, H * S2});
  auto VL_V = GetVLAPtr<T>(t_VL_V, {N, S2 * H});
  auto VL_TV = GetVLAPtr<T>(t_VL_TV, {N, H * S2});
  auto AP = GetVLAPtr<T>(t_AP, {SS1, S2 * S2});
  auto APD = GetVLAPtr<T>(t_APD, {SS1, S2 * S2});
  auto APD_T = GetVLAPtr<T>(t_APD_T, {SS1, S2 * S2}); // For BWD only
  auto APD_mask = GetVLAPtr<short>(t_APD_mask, {SS1, (S2 * S2 + 15) / 16});
  auto CL = GetVLAPtr<T>(t_CL, {N, S2 * H});
  auto HS = GetVLAPtr<T>(t_HS, {N, S2 * H});
  auto HS_T = GetVLAPtr<T>(t_HS_T, {S1, H * S2}); // for BWD only
  auto EHS = GetVLAPtr<T>(t_EHS, {N, S2 * H});
  auto EHS_T = GetVLAPtr<T>(t_EHS_T, {S1, H * S2}); // for BWD only
  auto AM = GetVLAPtr<T>(t_AM, {S2});
  auto offs = t_offs.data_ptr<long>();
  auto offs2 = t_offs2.data_ptr<long>();

  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(S2, H), BIAS);
  auto qkv_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2, H, H, S2 * H, H * H, 1.0, XformTPP::XFORM_NONE_TPP, 0, N)));
  auto xpose_tpp =
      SCOPEIT(XformExtTPP<T>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
  auto k_xpose_tpp_1 = SCOPEIT(
      XformExtTPP<T>(
          S2,
          H,
          training ? XformTPP::XFORM_N2V_TPP : XformTPP::XFORM_XPOSE_N2V_TPP,
          true),
      XPOSE);
  auto kv_xpose_tpp_2 =
      SCOPEIT(XformExtTPP<T>(S2, H, XformTPP::XFORM_XPOSE_N2V_TPP, true), VNNI);
  auto v_xpose_tpp_1 =
      SCOPEIT(XformExtTPP<T>(S2, H, XformTPP::XFORM_N2V_TPP, true), VNNI);
  auto a_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, float>(
      S2, S2, H, S2 * H, H * S2, 0.0, XformTPP::XFORM_NONE_TPP, 0, 1)));
  auto scale_tpp = SCOPEIT((ScaleTPP<float, float>(S2 * S2)), EW_SCL);
  auto add_mask_tpp = SCOPEIT(AddBiasTPP<T>(S2, S2), EW_ADD);
  auto softmax_fwd_tpp = SCOPEIT((VarSoftMaxFwdTPP<float, T>(S2, S2)), SOFTMAX);
  auto dropout_fwd_tpp = SCOPEIT(DropOutFwdTPP<T>(S2 * S2, p), DROPOUT);
  auto a_xpose_tpp =
      SCOPEIT(XformExtTPP<T>(S2, S2, XformTPP::XFORM_XPOSE_TPP), XPOSE);
  auto c_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2, H, S2, S2 * S2, N * S2 * H, 0.0, XformTPP::XFORM_NONE_TPP, 0, S1)));

  {
    RECORD_SCOPE(q_gemm, {t_HS, t_Wq_V});
    {
      long BN = N;
#ifdef NO_PARLOOPER
      for (int bn = 0; bn < N; bn += BN) {
#pragma omp parallel for collapse(2)
        for (int s1 = 0; s1 < S1; s1++) {
          for (int nk = 0; nk < N; nk++) {
            if (low_prec_training && nk == 0)
              xpose_tpp(BN, S2 * H, S1 * S2 * H, HS[s1][bn], HS_T[bn][s1]);
            if (bn == 0)
              copy_bias_tpp(Bq[nk], QL[s1][nk]);
            qkv_gemm_tpp(HS[s1][bn], Wq_V[nk][bn], QL[s1][nk], BN);
            if (low_prec_training)
              if (bn == N - BN)
                xpose_tpp(QL[s1][nk], QL_T[nk][s1]);
          }
        }
      }
#else
      auto loop_scheme = large_cache_opt ? "acB" : "aBC";
      // auto qkv_loop =
      //    ThreadedLoop<3>({{0L, N, BN, false}, {S1}, {N}}, loop_scheme);
      auto qkv_loop = ThreadedLoop<3>(
          {LoopSpecs{0L, N, BN, false}, LoopSpecs{S1}, LoopSpecs{N}},
          loop_scheme);
      qkv_loop(
          [&](int* ind) {
            int bn = ind[0], s1 = ind[1], nk = ind[2];
            if (low_prec_training && nk == 0)
              xpose_tpp(BN, S2 * H, S1 * S2 * H, HS[s1][bn], HS_T[bn][s1]);
            if (bn == 0)
              copy_bias_tpp(Bq[nk], QL[s1][nk]);
            qkv_gemm_tpp(HS[s1][bn], Wq_V[nk][bn], QL[s1][nk], BN, true);
            if (low_prec_training)
              if (bn == N - BN)
                xpose_tpp(QL[s1][nk], QL_T[nk][s1]);
          },
          [&]() { qkv_gemm_tpp.config(); },
          [&]() { qkv_gemm_tpp.release(); });
#endif
    }
  }

  {
    RECORD_SCOPE(k_gemm, {t_EHS, t_Wk_V});
    {
#ifdef NO_PARLOOPER
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int nk = 0; nk < N; nk++) {
          T tmp[S2 * H];
          T* tmpp = (training && !low_prec_training) ? KL_V[s1][nk] : tmp;
          if (!null_EHS && low_prec_training && nk == 0)
            xpose_tpp(N, S2 * H, S1 * S2 * H, EHS[s1][0], EHS_T[0][s1]);
          copy_bias_tpp(Bk[nk], tmpp);
          qkv_gemm_tpp(EHS[s1][0], Wk_V[nk][0], tmpp, N);
          k_xpose_tpp_1(tmpp, KL_V[s1][nk]); // KL_V = KL_VT if not training
          if (training)
            kv_xpose_tpp_2(tmpp, KL_TV[s1][nk]);
        }
      }
#else
      auto loop_scheme = large_cache_opt ? "bA" : "AB";
      // auto qkv_loop = ThreadedLoop<2>({{S1}, {N}}, loop_scheme);
      auto qkv_loop =
          ThreadedLoop<2>({LoopSpecs{S1}, LoopSpecs{N}}, loop_scheme);
      qkv_loop(
          [&](int* ind) {
            int s1 = ind[0], nk = ind[1];

            T tmp[S2 * H];
            T* tmpp = (training && !low_prec_training) ? KL_V[s1][nk] : tmp;
            if (!null_EHS && low_prec_training && nk == 0)
              xpose_tpp(N, S2 * H, S1 * S2 * H, EHS[s1][0], EHS_T[0][s1]);
            copy_bias_tpp(Bk[nk], tmpp);
            qkv_gemm_tpp(EHS[s1][0], Wk_V[nk][0], tmpp, N);
            k_xpose_tpp_1(tmpp, KL_V[s1][nk]); // KL_V = KL_VT if not training
            if (training)
              kv_xpose_tpp_2(tmpp, KL_TV[s1][nk]);
          },
          [&]() { qkv_gemm_tpp.config(); },
          [&]() { qkv_gemm_tpp.release(); });
#endif
    }
  }

  {
    RECORD_SCOPE(v_gemm, {t_EHS, t_Wv_V});
    {
#ifdef NO_PARLOOPER
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int nk = 0; nk < N; nk++) {
          T tmp[S2 * H];
          T* tmpp = (!dt_low_prec) ? VL_V[s1][nk] : tmp;
          copy_bias_tpp(Bv[nk], tmpp);
          qkv_gemm_tpp(EHS[s1][0], Wv_V[nk][0], tmpp, N);
          v_xpose_tpp_1(tmpp, VL_V[s1][nk]);
          if (training)
            kv_xpose_tpp_2(tmpp, VL_TV[s1][nk]);
        }
      }
#else
      auto loop_scheme = large_cache_opt ? "bA" : "AB";
      // auto qkv_loop = ThreadedLoop<2>({{S1}, {N}}, loop_scheme);
      auto qkv_loop =
          ThreadedLoop<2>({LoopSpecs{S1}, LoopSpecs{N}}, loop_scheme);
      qkv_loop(
          [&](int* ind) {
            int s1 = ind[0], nk = ind[1];
            T tmp[S2 * H];
            T* tmpp = (!dt_low_prec) ? VL_V[s1][nk] : tmp;
            copy_bias_tpp(Bv[nk], tmpp);
            qkv_gemm_tpp(EHS[s1][0], Wv_V[nk][0], tmpp, N);
            v_xpose_tpp_1(tmpp, VL_V[s1][nk]);
            if (training)
              kv_xpose_tpp_2(tmpp, VL_TV[s1][nk]);
          },
          [&]() { qkv_gemm_tpp.config(); },
          [&]() { qkv_gemm_tpp.release(); });
#endif
    }
  }
  // Take the dot product between "query" and "key" to get the raw attention
  // scores.
  {
    RECORD_SCOPE(ac_gemm, {t_QL, t_KL_TV});
    {
#pragma omp parallel for collapse(2) schedule(static, 1)
      for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
          long start = offs[b];
          long ss1 = offs2[b];
          long end = offs[b + 1];
          long len = end - start;
          for (int s11 = start; s11 < end; s11++, ss1 += len) {
            float AS[len][S2][S2];
            for (int s21 = start; s21 < end; s21++) {
              long ls21 = s21 - start;
              a_gemm_tpp(QL[s11][n], KL_TV[s21][n], AS[ls21][0], 1);
              scale_tpp(AS[ls21][0], AS[ls21][0], one_by_sqrt_H);
              if (t_AM.numel() != 0)
                add_mask_tpp(AM[s21], AS[ls21][0]);
            }
            softmax_fwd_tpp(len, AS[0][0], AP[n][ss1]);
            if (p > 0) {
              for (int l = 0; l < len; l++) {
                dropout_fwd_tpp(
                    AP[n][ss1 + l],
                    rng_state,
                    APD[n][ss1 + l],
                    APD_mask[n][ss1 + l]);
              }
            }
            if (t_HM.numel() != 0) {
              // FIXME: shape of head mask is not correct here yet
              TPP_ASSERT(0, "t_HM used");
              // t_APD[b][s11][n] *= t_HM[b][s11][n];
            }
            if (training) {
              long l = s11 - start;
              long ss = offs2[b];
              // xpose S1xS1 part as well here to allow fix stride in GEMM in
              // bwd
              a_xpose_tpp(
                  len, S2 * S2, len * S2 * S2, APD[n][ss1], APD_T[n][ss + l]);
            }
            c_gemm_tpp(APD[n][ss1], VL_V[start][n], CL[s11][n], len);
          }
        }
      }
    }
  }
}
// auto t_APO = t_APD.permute({0, 2, 1, 4, 3, 5}).contiguous().view({B, N, S,
// S});
auto t_APO = t_APD;
return std::vector<at::Tensor>({t_CL,
                                t_APO,
                                t_HS_T,
                                null_EHS ? t_EHS_orig : t_EHS_T,
                                t_QL_T,
                                t_KL_V,
                                t_VL_TV,
                                t_AP,
                                t_APD_T,
                                t_APD_mask});

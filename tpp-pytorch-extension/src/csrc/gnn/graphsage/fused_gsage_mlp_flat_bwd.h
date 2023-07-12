/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION("gsage_mlp_bwd", std::vector<c10::IValue>());

at::Tensor t_in, t_in_res, t_wt, t_wt_res;
int i = 0;

const int threads = omp_get_max_threads();
auto t_grad_out = inputs[i++].contiguous();

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
auto t_relu_mask = inputs[i++];
auto t_dp_mask = inputs[i++];

auto in_sizes = t_in.sizes();
auto wt_sizes = t_wt.sizes();
auto N = in_sizes[0];
auto C = in_sizes[1];
auto bn = align;
auto nn = N / bn;
auto rem = N % bn;

auto nk = wt_sizes[0];
auto nc = wt_sizes[1];
auto bc = wt_sizes[2];
if (t_wt.dtype() == at::kBFloat16)
  bc = bc * wt_sizes[4];
auto bk = wt_sizes[3];

auto bnp = bn;
auto bkp = bk;
auto bcp = bc;
auto remp = rem;

auto K = nk * bk;

if (t_in.dtype() == at::kBFloat16) {
  bnp = bn + bn % 2;
  remp = rem + rem % 2;
}

if (t_wt.dtype() == at::kBFloat16) {
  bcp = bc + bc % 2;
  bkp = bk + bk % 2;
}

long rd = (bk + 15) / 16;
auto relu_mask = GetVLAPtr<short>(t_relu_mask, {bn, nk, rd});
auto dp_mask = GetVLAPtr<short>(t_dp_mask, {bn, nk, rd});

const auto input_trans_flag =
    (t_in.dtype() == at::kFloat ? XformTPP::XFORM_XPOSE_TPP
                                : XformTPP::XFORM_NONE_TPP);

auto t_wt_TV = wt_tensor_for_bwd(nk, bk, nc, bc, t_wt);

at::Tensor t_wt_res_TV = at::empty(0);
if (res)
  t_wt_res_TV = wt_tensor_for_bwd(nk, bk, nc, bc, t_wt_res);

auto t_grad_in = t_in.new_empty({N, C});
at::Tensor t_grad_in_res = at::empty(0);
if (res) {
  t_grad_in_res = t_in_res.new_empty({N, C});
}

auto t_grad_wt = at::empty_like(t_wt);
at::Tensor t_grad_wt_tmp;
if (t_wt.dtype() == at::kBFloat16)
  t_grad_wt_tmp = at::empty({nk, nc, bc, bk});
else
  t_grad_wt_tmp = t_grad_wt;

at::Tensor t_grad_wt_res = at::empty(0);
at::Tensor t_grad_wt_res_tmp = at::empty(0);
if (res) {
  t_grad_wt_res = at::empty_like(t_wt);
  if (t_wt.dtype() == at::kBFloat16)
    t_grad_wt_res_tmp = at::empty({nk, nc, bc, bk});
  else
    t_grad_wt_res_tmp = t_grad_wt_res;
}

auto t_grad_bias = t_wt.new_empty({nk * bk});

at::Tensor t_grad_out_f32 = t_grad_out;
if (t_grad_out.dtype() == at::kBFloat16)
  t_grad_out_f32 = at::empty({N, K});

auto grad_out = GetVLAPtr<T>(t_grad_out, {bn, nk, bk});
auto grad_out_f32 = GetVLAPtr<float>(t_grad_out_f32, {bn, nk, bk});

auto grad_in = GetVLAPtr<T>(t_grad_in, {bn, nc, bc});
auto grad_in_res = GetVLAPtr<T>(t_grad_in_res, {bn, nc, bc});

// del-weights and weights in blocked layout
auto grad_wt = GetVLAPtr<T>(t_grad_wt, {nc, bc* bk});
auto grad_wt_res = GetVLAPtr<T>(t_grad_wt_res, {nc, bc* bk});
auto grad_wt_tmp = GetVLAPtr<float>(t_grad_wt_tmp, {nc, bc* bk});
auto grad_wt_res_tmp = GetVLAPtr<float>(t_grad_wt_res_tmp, {nc, bc* bk});

auto wt_TV = GetVLAPtr<T>(t_wt_TV, {nc, bkp* bc});
auto wt_res_TV = GetVLAPtr<T>(t_wt_res_TV, {nc, bkp* bc});
auto grad_bias = GetVLAPtr<T>(t_grad_bias, {bk});

auto in = GetVLAPtr<T>(t_in, {bn, nc, bc}); // flat layout for fp32
auto in_res = GetVLAPtr<T>(t_in_res, {bn, nc, bc}); // flat layout for fp32

auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(nk * bk), EW_ZERO);
auto set_zero_col_tpp = SCOPEIT(SetZeroTPP<T>(bn, 1, bkp), EW_ZERO);
auto grad_bias_tpp = SCOPEIT(GradBiasTPP<float>(bn, bk, K), BIAS);
auto dropout_bwd_tpp =
    SCOPEIT((DropOutBwdTPP<T, float>(bn, bk, K, K, p)), DROPOUT);
auto relu_bwd_tpp = SCOPEIT(ReLUBwdTPP<float>(bn, bk, K, K, true), ACT);
auto n2v_tpp = SCOPEIT(
    XformExtTPP<T>(bn, bk, bnp, bk, nk* bk, bk, XformTPP::XFORM_N2V_TPP, true),
    VNNI);
auto n2v_wt_tpp = SCOPEIT(
    XformExtTPP<T>(bc, bk, bcp, bk, XformTPP::XFORM_N2V_TPP, true),
    VNNI);
auto cpy_tpp = SCOPEIT(CpyTPP<T>(bn, bk, bk, bkp), EW_COPY);
auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(bn, bk, K, K)), EW_COPY);
auto cvt_f32_tpp = SCOPEIT((ConvertTPP<T, float>(bn, bk, K, K)), EW_COPY);
auto add_gwt_tpp = SCOPEIT((AddTPP<float, float>(bc, bk)), EW_ADD);

auto brgemm_di_tpp = SCOPEIT(
    (BrgemmTPP<
        T,
        T>(bn, bc, bkp, bkp, nc* bc* bkp, nk* bkp, bc, nc* bc, 0.0, 0, nk)));

auto brgemm_dw_f32_tpp = SCOPEITGEMM2((BrgemmTPP<T, float>(
    bc,
    bk,
    bnp,
    C* bnp,
    K* bnp,
    C,
    K,
    bk,
    0.0,
    input_trans_flag,
    16)));
auto brgemm_dw_f32_tpp_b1 = SCOPEITGEMM2((BrgemmTPP<T, float>(
    bc,
    bk,
    bnp,
    C* bnp,
    K* bnp,
    C,
    K,
    bk,
    1.0,
    input_trans_flag,
    16)));

// BF16 del-wt brgemms
auto brgemm_dw_bf16_tpp =
    SCOPEIT((BrgemmTPP<
             T,
             float>(bc, bk, bnp, bc* bnp, bk* bnp, bnp, bk, bk, 0.0, 0, 16)));
auto brgemm_dw_bf16_tpp_b1 =
    SCOPEIT((BrgemmTPP<
             T,
             float>(bc, bk, bnp, bc* bnp, bk* bnp, bnp, bk, bk, 1.0, 0, 16)));

if (apply_bias) {
  RECORD_SCOPE(gdbias, {t_grad_out});
  {
    tensor_set_zero(nk, bk, t_grad_bias);
    float* bias_ptrs[threads];
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        float prv_grad_bias[nk][bk];
        bias_ptrs[tid] = prv_grad_bias[0];
        set_zero_tpp(prv_grad_bias[0]);
#pragma omp for collapse(2)
        for (int n = 0; n < nn; n++) {
          for (int k = 0; k < nk; k++) {
            if (p > 0) {
              dropout_bwd_tpp(
                  grad_out[n][0][k], grad_out_f32[n][0][k], dp_mask[n][0][k]);
            } else
              cvt_f32_tpp(grad_out[n][0][k], grad_out_f32[n][0][k]);

            if (act == "relu") {
              relu_bwd_tpp(
                  grad_out_f32[n][0][k],
                  grad_out_f32[n][0][k],
                  (float*)NULL,
                  relu_mask[n][0][k]);
            }
            grad_bias_tpp(grad_out_f32[n][0][k], prv_grad_bias[k]);
            if (p > 0 || act == "relu")
              cvt_tpp(grad_out_f32[n][0][k], grad_out[n][0][k]);
          }
        }
#pragma omp barrier
        omp_reduce_buf(threads, nk * bk, bias_ptrs, grad_bias[0]);
      }
      if (rem > 0) {
        auto grad_out = GetVLAPtr<T>(t_grad_out, {nk, bk});
        auto grad_out_f32 = GetVLAPtr<float>(t_grad_out_f32, {nk, bk});
        auto relu_mask = GetVLAPtr<short>(t_relu_mask, {nk, rd});
        auto dp_mask = GetVLAPtr<short>(t_dp_mask, {nk, rd});

        auto dropout_bwd_tpp =
            SCOPEIT((DropOutBwdTPP<T, float>(1, bk, K, K, p)), DROPOUT);
        auto relu_bwd_tpp = SCOPEIT(ReLUBwdTPP<float>(1, bk, K, K, true), ACT);
        auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(1, bk, K, K)), EW_COPY);
        auto cvt_f32_tpp =
            SCOPEIT((ConvertTPP<T, float>(1, bk, K, K)), EW_COPY);
        auto grad_bias_tpp = SCOPEIT(GradBiasTPP<float>(1, bk, K), BIAS);

        float prv_grad_bias[nk][bk];
        bias_ptrs[0] = prv_grad_bias[0];
        set_zero_tpp(prv_grad_bias[0]);

        for (int k = 0; k < nk; k++) {
          for (int r = 0; r < rem; r++) {
            if (p > 0) {
              dropout_bwd_tpp(
                  grad_out[nn * bn + r][k],
                  grad_out_f32[nn * bn + r][k],
                  dp_mask[nn * bn + r][k]);
            } else
              cvt_f32_tpp(
                  grad_out[nn * bn + r][k], grad_out_f32[nn * bn + r][k]);
            if (act == "relu") {
              relu_bwd_tpp(
                  grad_out_f32[nn * bn + r][k],
                  grad_out_f32[nn * bn + r][k],
                  (float*)NULL,
                  relu_mask[nn * bn + r][k]);
            }
            grad_bias_tpp(grad_out_f32[nn * bn + r][k], prv_grad_bias[k]);
            if (p > 0 || act == "relu")
              cvt_tpp(grad_out_f32[nn * bn + r][k], grad_out[nn * bn + r][k]);
          }
        }
        omp_reduce_buf(1, nk * bk, bias_ptrs, grad_bias[0], true);
      }
    }
  }
} else {
  RECORD_SCOPE(gdout, {t_grad_out});
  {
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int n = 0; n < nn; n++) {
        for (int k = 0; k < nk; k++) {
          if (p > 0) {
            dropout_bwd_tpp(
                grad_out[n][0][k], grad_out_f32[n][0][k], dp_mask[n][0][k]);
          } else
            cvt_f32_tpp(grad_out[n][0][k], grad_out_f32[n][0][k]);

          if (act == "relu") {
            relu_bwd_tpp(
                grad_out_f32[n][0][k],
                grad_out_f32[n][0][k],
                (float*)NULL,
                relu_mask[n][0][k]);
          }
          if (p > 0 || act == "relu")
            cvt_tpp(grad_out_f32[n][0][k], grad_out[n][0][k]);
        }
      }
      if (rem > 0) {
        auto grad_out = GetVLAPtr<T>(t_grad_out, {nk, bk});
        auto grad_out_f32 = GetVLAPtr<float>(t_grad_out_f32, {nk, bk});
        auto relu_mask = GetVLAPtr<short>(t_relu_mask, {nk, rd});
        auto dp_mask = GetVLAPtr<short>(t_dp_mask, {nk, rd});

        auto dropout_bwd_tpp =
            SCOPEIT((DropOutBwdTPP<T, float>(1, bk, K, K, p)), DROPOUT);
        auto relu_bwd_tpp = SCOPEIT(ReLUBwdTPP<float>(1, bk, K, K, true), ACT);
        auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(1, bk, K, K)), EW_COPY);
        auto cvt_f32_tpp =
            SCOPEIT((ConvertTPP<T, float>(1, bk, K, K)), EW_COPY);
        auto grad_bias_tpp = SCOPEIT(GradBiasTPP<float>(1, bk, K), BIAS);

        for (int k = 0; k < nk; k++) {
          for (int r = 0; r < rem; r++) {
            if (p > 0) {
              dropout_bwd_tpp(
                  grad_out[nn * bn + r][k],
                  grad_out_f32[nn * bn + r][k],
                  dp_mask[nn * bn + r][k]);
            } else
              cvt_f32_tpp(
                  grad_out[nn * bn + r][k], grad_out_f32[nn * bn + r][k]);
            if (act == "relu") {
              relu_bwd_tpp(
                  grad_out_f32[nn * bn + r][k],
                  grad_out_f32[nn * bn + r][k],
                  (float*)NULL,
                  relu_mask[nn * bn + r][k]);
            }
            if (p > 0 || act == "relu")
              cvt_tpp(grad_out_f32[nn * bn + r][k], grad_out[nn * bn + r][k]);
          }
        }
      }
    }
  }
}

{
  RECORD_SCOPE(gdi_gemm, {t_grad_out, t_wt});
  {
    if (bk != bkp) {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
      {
        brgemm_di_tpp.config();

        T tmp[bn][nk * bkp];
        for (int k = 0; k < nk; k++)
          set_zero_col_tpp(&tmp[0][k * bkp] + bk);

        int tid = omp_get_thread_num();
        int threads = omp_get_num_threads();
        int work = nn * nc;
        int chunk =
            (work % threads == 0) ? (work / threads) : (work / threads) + 1;
        int chunk_start = (tid * chunk < work) ? (tid * chunk) : work;
        int chunk_end = ((tid + 1) * chunk < work) ? ((tid + 1) * chunk) : work;

        for (int n3c = chunk_start; n3c < chunk_end; n3c++) {
          int n = n3c / nc;
          int c = n3c % nc;

          for (int k = 0; k < nk; k++)
            cpy_tpp(grad_out[n][0][k], &tmp[0][k * bk]);

          brgemm_di_tpp(tmp[0], wt_TV[0][c], grad_in[n][0][c], nk, true);
          if (res)
            brgemm_di_tpp(
                tmp[0], wt_res_TV[0][c], grad_in_res[n][0][c], nk, true);
        }
        brgemm_di_tpp.release();
      }
    } else {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        int threads = omp_get_num_threads();
        int work = nn * nc;
        int chunk =
            (work % threads == 0) ? (work / threads) : (work / threads) + 1;
        int chunk_start = (tid * chunk < work) ? (tid * chunk) : work;
        int chunk_end = ((tid + 1) * chunk < work) ? ((tid + 1) * chunk) : work;

        brgemm_di_tpp.config();

        for (int n3c = chunk_start; n3c < chunk_end; n3c++) {
          int n = n3c / nc;
          int c = n3c % nc;
          brgemm_di_tpp(
              grad_out[n][0][0], wt_TV[0][c], grad_in[n][0][c], nk, true);
          if (res)
            brgemm_di_tpp(
                grad_out[n][0][0],
                wt_res_TV[0][c],
                grad_in_res[n][0][c],
                nk,
                true);
        }
        brgemm_di_tpp.release();
      }
    }
    if (rem > 0) {
      auto grad_out = GetVLAPtr<T>(t_grad_out, {nk, bk});
      auto grad_in = GetVLAPtr<T>(t_grad_in, {nc, bc});
      auto grad_in_res = GetVLAPtr<T>(t_grad_in_res, {nc, bc});

      auto set_zero_col_tpp = SCOPEIT(SetZeroTPP<T>(rem, 1, bkp), EW_ZERO);
      auto cpy_tpp = SCOPEIT(CpyTPP<T>(rem, bk, bk, bkp), EW_COPY);
      auto brgemm_di_tpp = SCOPEIT((BrgemmTPP<T, T>(
          rem,
          bc,
          bkp,
          bkp,
          nc * bc * bkp,
          nk * bkp,
          bc,
          nc * bc,
          0.0,
          0,
          nk)));

      brgemm_di_tpp.config();

      if (bk != bkp) {
        T tmp[rem][nk * bkp];

        for (int k = 0; k < nk; k++) {
          set_zero_col_tpp(&tmp[0][k * bk] + bk);
          cpy_tpp(grad_out[nn * bn][0], &tmp[0][k * bk]);
        }
        for (int c = 0; c < nc; c++)
          brgemm_di_tpp(tmp[0], wt_TV[0][c], grad_in[nn * bn][c], nk, true);
        if (res)
          for (int c = 0; c < nc; c++)
            brgemm_di_tpp(
                tmp[0], wt_res_TV[0][c], grad_in_res[nn * bn][c], nk, true);

      } else {
        for (int c = 0; c < nc; c++)
          brgemm_di_tpp(
              grad_out[nn * bn][0], wt_TV[0][c], grad_in[nn * bn][c], nk, true);

        if (res) {
          for (int c = 0; c < nc; c++)
            brgemm_di_tpp(
                grad_out[nn * bn][0],
                wt_res_TV[0][c],
                grad_in_res[nn * bn][c],
                nk,
                true);
        }
      }
      brgemm_di_tpp.release();
    }
  }
}

auto trans_tpp = SCOPEIT(
    XformExtTPP<
        T>(bn, bc, bc, bnp, nc* bc, bnp, XformTPP::XFORM_XPOSE_TPP, true),
    XPOSE);
{
  RECORD_SCOPE(gdw_gemm, {t_in, t_grad_out});
  {
    int upd_n_weight_copies;
    int BF;
#if 1
    upd_n_weight_copies = nk * nc < 4 * threads ? threads : 1;
    BF = 32;
#else
    BF = atoi(getenv("BF"));
    upd_n_weight_copies = atoi(getenv("UPD_WEIGHT_COPIES"));
#endif
    const int fm_blocking = (bk % 16 == 0) ? 16 : bk;
    const int reduce_work = nk * nc * (bk / fm_blocking) * bc;
    const int reduce_chunksize = (reduce_work % threads == 0)
        ? (reduce_work / threads)
        : (reduce_work / threads) + 1;
    const int chunk0 = reduce_chunksize * fm_blocking;
    const int chunk1 =
        (reduce_work - (reduce_work / reduce_chunksize) * reduce_chunksize) *
        fm_blocking;

    int blocks_per_layer = (nn + upd_n_weight_copies - 1) / upd_n_weight_copies;
    int reduce_rows = (nn % blocks_per_layer == 0)
        ? (nn / blocks_per_layer)
        : (nn / blocks_per_layer) + 1;

    auto wt_reduce_chunk0_tpp = SCOPEIT(
        (ReduceAddColTPP<float, float>(reduce_rows, chunk0, K * C, chunk0)),
        EW_RED);
    auto wt_reduce_chunk1_tpp = SCOPEIT(
        (ReduceAddColTPP<float, float>(reduce_rows, chunk1, K * C, chunk1)),
        EW_RED);
    auto setzero_delwt_tpp = SCOPEIT(SetZeroTPP<float>(bc * bk), EW_ZERO);

    at::Tensor t_grad_wt_priv =
        at::empty({upd_n_weight_copies, nk, nc, bc * bk});
    at::Tensor t_grad_wt_res_priv =
        at::empty({upd_n_weight_copies, nk, nc, bc * bk});
    auto grad_wt_priv = GetVLAPtr<float>(t_grad_wt_priv, {nk, nc, bc * bk});
    auto grad_wt_res_priv =
        GetVLAPtr<float>(t_grad_wt_res_priv, {nk, nc, bc * bk});

    at::Tensor t_global_tmp_go = at::empty(0);
    at::Tensor t_global_tmp_inT = at::empty(0);
    at::Tensor t_global_tmp_in_resT = at::empty(0);

    if (t_grad_out.dtype() == at::kBFloat16 && t_wt.dtype() == at::kBFloat16) {
      t_global_tmp_go =
          at::empty({threads, (nn / BF + 1), bnp * bk}, at::kBFloat16);
      t_global_tmp_inT =
          at::empty({threads, nc, (nn / BF + 1), bnp * bc}, at::kBFloat16);
      t_global_tmp_in_resT =
          at::empty({threads, nc, (nn / BF + 1), bnp * bc}, at::kBFloat16);
    }
    auto global_tmp_go =
        GetVLAPtr<T>(t_global_tmp_go, {(nn / BF + 1), bnp * bk});
    auto global_tmp_inT =
        GetVLAPtr<T>(t_global_tmp_inT, {nc, (nn / BF + 1), bnp * bc});
    auto global_tmp_in_resT =
        GetVLAPtr<T>(t_global_tmp_in_resT, {nc, (nn / BF + 1), bnp * bc});

    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned long long blocks;
      int team_id = tid / (threads / upd_n_weight_copies);
      int in_team_id = tid % (threads / upd_n_weight_copies);
      int mb_blocks_chunksize = (nn % upd_n_weight_copies == 0)
          ? (nn / upd_n_weight_copies)
          : ((nn / upd_n_weight_copies) + 1);
      const int my_mb_start = (team_id * mb_blocks_chunksize < nn)
          ? (team_id * mb_blocks_chunksize)
          : nn;
      const int my_mb_end = ((team_id + 1) * mb_blocks_chunksize < nn)
          ? ((team_id + 1) * mb_blocks_chunksize)
          : nn;
      const int my_mb_blocks = my_mb_end - my_mb_start;
      const int threads_per_team = threads / upd_n_weight_copies;
      int ifm_chunk = (nc % threads_per_team == 0) ? nc / threads_per_team
                                                   : nc / threads_per_team + 1;
      int my_ifm_start =
          (in_team_id * ifm_chunk < nc) ? in_team_id * ifm_chunk : nc;
      int my_ifm_end = ((in_team_id + 1) * ifm_chunk < nc)
          ? (in_team_id + 1) * ifm_chunk
          : nc;
      int mb_block_step = (my_mb_blocks % BF == 0) ? (my_mb_blocks / BF)
                                                   : ((my_mb_blocks / BF) + 1);

      if (t_grad_out.dtype() == at::kBFloat16)
        brgemm_dw_bf16_tpp_b1.config();

      for (int bfn = my_mb_start; bfn < my_mb_end; bfn += mb_block_step) {
        blocks = (bfn + mb_block_step <= my_mb_end) ? mb_block_step
                                                    : my_mb_end - bfn;
        for (int ofm1 = 0; ofm1 < nk; ++ofm1) {
          if (t_in.dtype() == at::kBFloat16) {
            n2v_tpp(
                blocks,
                K * bnp,
                bk * bnp,
                grad_out[bfn][0][ofm1],
                global_tmp_go[tid][0]);
          }
          for (int ifm1 = my_ifm_start; ifm1 < my_ifm_end; ++ifm1) {
            if (bfn == my_mb_start) {
              /* initiaize current work task to zero */
              setzero_delwt_tpp(grad_wt_priv[team_id][ofm1][ifm1]);
              setzero_delwt_tpp(grad_wt_res_priv[team_id][ofm1][ifm1]);
            }
            if (t_in.dtype() == at::kFloat) {
              brgemm_dw_f32_tpp_b1(
                  in[bfn][0][ifm1],
                  grad_out[bfn][0][ofm1],
                  grad_wt_priv[team_id][ofm1][ifm1],
                  blocks);
              if (res) {
                brgemm_dw_f32_tpp_b1(
                    in_res[bfn][0][ifm1],
                    grad_out[bfn][0][ofm1],
                    grad_wt_res_priv[team_id][ofm1][ifm1],
                    blocks);
              }
            } else if (t_in.dtype() == at::kBFloat16) {
              if (ofm1 == 0)
                trans_tpp(
                    blocks,
                    C * bnp,
                    bc * bnp,
                    in[bfn][0][ifm1],
                    global_tmp_inT[tid][ifm1 - my_ifm_start][0]);

              brgemm_dw_bf16_tpp_b1(
                  global_tmp_inT[tid][ifm1 - my_ifm_start][0],
                  global_tmp_go[tid][0],
                  grad_wt_priv[team_id][ofm1][ifm1],
                  blocks,
                  true);

              if (res) {
                if (ofm1 == 0)
                  trans_tpp(
                      blocks,
                      C * bnp,
                      bc * bnp,
                      in_res[bfn][0][ifm1],
                      global_tmp_in_resT[tid][ifm1 - my_ifm_start][0]);
                brgemm_dw_bf16_tpp_b1(
                    global_tmp_in_resT[tid][ifm1 - my_ifm_start][0],
                    global_tmp_go[tid][0],
                    grad_wt_res_priv[team_id][ofm1][ifm1],
                    blocks,
                    true);
              }
            }
          }
        }
      }

      if (t_grad_out.dtype() == at::kBFloat16)
        brgemm_dw_bf16_tpp_b1.release();

      const int reduce_thr_begin = (tid * reduce_chunksize < reduce_work)
          ? (tid * reduce_chunksize)
          : reduce_work;
      const int reduce_thr_end = ((tid + 1) * reduce_chunksize < reduce_work)
          ? ((tid + 1) * reduce_chunksize)
          : reduce_work;
#pragma omp barrier
      float* in = grad_wt_priv[0][0][0] + reduce_thr_begin * fm_blocking;
      float* out = grad_wt_tmp[0][0] + reduce_thr_begin * fm_blocking;
      float *in_res, *out_res;
      if (res) {
        in_res = grad_wt_res_priv[0][0][0] + reduce_thr_begin * fm_blocking;
        out_res = grad_wt_res_tmp[0][0] + reduce_thr_begin * fm_blocking;
      }
      if ((reduce_thr_end - reduce_thr_begin) == reduce_chunksize) {
        wt_reduce_chunk0_tpp(in, out);
        if (res) {
          wt_reduce_chunk0_tpp(in_res, out_res);
        }
      } else {
        if ((reduce_thr_end - reduce_thr_begin) > 0) {
          wt_reduce_chunk1_tpp(in, out);
          if (res) {
            wt_reduce_chunk1_tpp(in_res, out_res);
          }
        }
      }
    }
    if (rem > 0) {
      auto grad_out = GetVLAPtr<T>(t_grad_out, {nk, bk});
      auto in = GetVLAPtr<T>(t_in, {nc, bc});
      auto in_res = GetVLAPtr<T>(t_in_res, {nc, bc});
      auto brgemm_dw_f32_tpp_b1 = SCOPEITGEMM2((BrgemmTPP<T, float>(
          bc,
          bk,
          remp,
          C * remp,
          K * remp,
          C,
          K,
          bk,
          1.0,
          input_trans_flag,
          1)));
      auto brgemm_dw_bf16_tpp_b1 = SCOPEIT((BrgemmTPP<T, float>(
          bc, bk, remp, bc * remp, bk * remp, remp, bk, bk, 1.0, 0, 1)));
      auto n2v_tpp = SCOPEIT(
          XformExtTPP<T>(
              rem, bk, remp, bk, nk * bk, bk, XformTPP::XFORM_N2V_TPP, true),
          VNNI);
      auto trans_tpp = SCOPEIT(
          XformExtTPP<T>(
              rem,
              bc,
              bc,
              remp,
              nc * bc,
              remp,
              XformTPP::XFORM_XPOSE_TPP,
              true),
          XPOSE);

      if (t_wt.dtype() == at::kFloat) {
#pragma omp parallel for collapse(2)
        for (int k = 0; k < nk; k++) {
          for (int c = 0; c < nc; c++) {
            brgemm_dw_f32_tpp_b1(
                in[nn * bn][c], grad_out[nn * bn][k], grad_wt_tmp[k][c], 1);
            if (res)
              brgemm_dw_f32_tpp_b1(
                  in_res[nn * bn][c],
                  grad_out[nn * bn][k],
                  grad_wt_res_tmp[k][c],
                  1);
          }
        }
      } else if (t_wt.dtype() == at::kBFloat16) {
        T tmp_go[remp * bk], tmp_inT[remp * bc];
#pragma omp parallel
        {
          int tid = omp_get_thread_num();
          int threads = omp_get_num_threads();
          int work = nk * nc;
          int chunk =
              (work % threads == 0) ? (work / threads) : (work / threads) + 1;
          int chunk_start = (tid * chunk < work) ? (tid * chunk) : work;
          int chunk_end =
              ((tid + 1) * chunk < work) ? ((tid + 1) * chunk) : work;

          brgemm_dw_bf16_tpp_b1.config();

          for (int kk = chunk_start; kk < chunk_end; kk++) {
            int k = kk / nc;
            int c = kk % nc;

            n2v_tpp(grad_out[nn * bn][k], tmp_go);
            trans_tpp(in[nn * bn][c], tmp_inT);
            brgemm_dw_bf16_tpp_b1(tmp_inT, tmp_go, grad_wt_tmp[k][c], 1, true);
            if (res) {
              trans_tpp(in_res[nn * bn][c], tmp_inT);
              brgemm_dw_bf16_tpp_b1(
                  tmp_inT, tmp_go, grad_wt_res_tmp[k][c], 1, true);
            }
          }
          brgemm_dw_bf16_tpp_b1.release();
        }
      }
    }
    if (t_wt.dtype() == at::kBFloat16) {
#pragma omp parallel for collapse(2)
      for (int k = 0; k < nk; k++) {
        for (int c = 0; c < nc; c++) {
          n2v_wt_tpp(grad_wt_tmp[k][c], grad_wt[k][c]);
          if (res)
            n2v_wt_tpp(grad_wt_res_tmp[k][c], grad_wt_res[k][c]);
        }
      }
    }
  }
}

if (res) {
  return {t_grad_in, t_grad_in_res, t_grad_wt, t_grad_wt_res, t_grad_bias};
} else {
  return {t_grad_in, t_grad_wt, t_grad_bias};
}

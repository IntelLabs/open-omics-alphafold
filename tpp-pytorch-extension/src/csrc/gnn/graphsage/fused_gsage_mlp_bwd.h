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

auto t_grad_out = inputs[i++].contiguous();

if (res) {
  t_in = inputs[i++];
  t_in_res = inputs[i++];
  t_wt = inputs[i++];
  t_wt_res = inputs[i++];
} else {
  t_in = inputs[i++];
  t_wt = inputs[i++];
}
auto t_relu_mask = inputs[i++];
auto t_dp_mask = inputs[i++];

auto in_sizes = t_in.sizes();
auto wt_sizes = t_wt.sizes();
auto nn = in_sizes[0];
auto nc = in_sizes[1];
auto bn = in_sizes[2];
auto bc = in_sizes[3];

auto nk = wt_sizes[0];
auto bk = wt_sizes[3];

auto bnp = bn;
auto bkp = bk;
auto bcp = bc;

if (t_in.dtype() == at::kBFloat16) {
  bnp = bn + bn % 2;
}

if (t_wt.dtype() == at::kBFloat16) {
  bcp = bc + bc % 2;
  bkp = bk + bk % 2;
}

int rd = (bn * bk + 15) / 16;
auto relu_mask = GetVLAPtr<short>(t_relu_mask, {nk, rd});
auto dp_mask = GetVLAPtr<short>(t_dp_mask, {nk, rd});

const auto grad_wt_flag =
    (t_wt.dim() == 5 ? XformTPP::XFORM_N2V_TPP : XformTPP::XFORM_NONE_TPP);
const auto input_trans_flag =
    (t_in.dtype() == at::kFloat ? XformTPP::XFORM_XPOSE_TPP
                                : XformTPP::XFORM_NONE_TPP);

auto t_wt_TV = wt_tensor_for_bwd(nk, bk, nc, bc, t_wt);

at::Tensor t_wt_res_TV;
if (res)
  t_wt_res_TV = wt_tensor_for_bwd(nk, bk, nc, bc, t_wt_res);

auto t_in_T = t_in;
if (input_trans_flag == XformTPP::XFORM_NONE_TPP) {
  t_in_T = t_in.new_empty({nn, nc, bnp, bc});
  auto in_T = GetVLAPtr<bfloat16>(t_in_T, {bnp * bc});
  auto in = GetVLAPtr<bfloat16>(t_in, {bn * bc});
  auto trans_tpp =
      SCOPEIT(XformExtTPP<bfloat16>(bn, bc, XformTPP::XFORM_XPOSE_TPP), XPOSE);
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < nn * nc; n++) {
      trans_tpp(in[n], in_T[n]);
    }
  }
}

at::Tensor t_in_res_T = t_in_res;
if (res) {
  if (input_trans_flag == XformTPP::XFORM_NONE_TPP) {
    t_in_res_T = t_in_res.new_empty({nn, nc, bnp, bc});
    auto in_res_T = GetVLAPtr<bfloat16>(t_in_res_T, {bnp * bc});
    auto in_res = GetVLAPtr<bfloat16>(t_in_res, {bn * bc});
    auto trans_tpp = SCOPEIT(
        XformExtTPP<bfloat16>(bnp, bc, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < nn * nc; n++) {
        trans_tpp(in_res[n], in_res_T[n]);
      }
    }
  }
}

auto t_grad_in = t_in.new_empty({nn, nc, bn, bc});
at::Tensor t_grad_in_res;
if (res) {
  t_grad_in_res = t_in_res.new_empty({nn, nc, bn, bc});
}

auto t_grad_wt = at::empty_like(t_wt);
auto t_grad_wt_tmp = t_grad_wt;
if (t_wt.dtype() == at::kBFloat16)
  t_grad_wt_tmp = at::empty({nk, nc, bc, bk}, at::kBFloat16);

at::Tensor t_grad_wt_res, t_grad_wt_res_tmp;
if (res) {
  t_grad_wt_res = at::empty_like(t_wt);
  t_grad_wt_res_tmp = t_grad_wt_res;
  if (t_wt.dtype() == at::kBFloat16)
    t_grad_wt_res_tmp = at::empty({nk, nc, bc, bk}, at::kBFloat16);
}

auto t_grad_bias = t_wt.new_empty({nk * bk});

auto t_grad_out_V = t_grad_out;
if (t_grad_out.dtype() == at::kBFloat16)
  t_grad_out_V = t_grad_out.new_empty({nn, nk, bnp / 2, bk, 2});

auto t_grad_out_K = t_grad_out;
if (t_grad_out.dtype() == at::kBFloat16) {
  if (bk != bkp)
    t_grad_out_K = t_grad_out.new_empty({nn, nk, bn, bkp});
}

auto t_grad_out_f32 = t_grad_out;
;
if (t_grad_out.dtype() == at::kBFloat16)
  t_grad_out_f32 = at::empty({nn, nk, bn, bk});

auto grad_out = GetVLAPtr<T>(t_grad_out, {nk, bn* bk});
auto grad_out_K = GetVLAPtr<T>(t_grad_out_K, {nk, bn* bkp});
auto grad_out_V = GetVLAPtr<T>(t_grad_out_V, {nk, bnp* bk});
auto grad_out_f32 = GetVLAPtr<float>(t_grad_out_f32, {nk, bn* bk});
auto grad_in = GetVLAPtr<T>(t_grad_in, {nc, bn* bc});
auto grad_in_res = GetVLAPtr<T>(t_grad_in_res, {nc, bn* bc});
auto grad_wt = GetVLAPtr<T>(t_grad_wt, {nc, bcp* bk});
auto grad_wt_res = GetVLAPtr<T>(t_grad_wt_res, {nc, bcp* bk});
auto grad_wt_tmp = GetVLAPtr<T>(t_grad_wt_tmp, {nc, bc* bk});
auto grad_wt_res_tmp = GetVLAPtr<T>(t_grad_wt_res_tmp, {nc, bc* bk});
auto in_T = GetVLAPtr<T>(t_in_T, {nc, bnp* bc});
auto in_res_T = GetVLAPtr<T>(t_in_res_T, {nc, bnp* bc});
auto wt_TV = GetVLAPtr<T>(t_wt_TV, {nc, bkp* bc});
auto wt_res_TV = GetVLAPtr<T>(t_wt_res_TV, {nc, bkp* bc});
auto grad_bias = GetVLAPtr<T>(t_grad_bias, {bk});

auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(nk * bk), EW_ZERO);
auto set_zero_col_tpp = SCOPEIT(SetZeroTPP<T>(bn, 1, bkp), EW_ZERO);
auto grad_bias_tpp = SCOPEIT(GradBiasTPP<float>(nk, bk), BIAS);
auto dropout_bwd_tpp = SCOPEIT((DropOutBwdTPP<T, float>(bn * bk, p)), DROPOUT);
auto relu_bwd_tpp = SCOPEIT(ReLUBwdTPP<float>(bn * bk), ACT);
auto n2v_tpp = SCOPEIT(
    XformExtTPP<T>(bn, bk, bnp, bk, XformTPP::XFORM_N2V_TPP, true),
    VNNI);
auto n2v_wt_tpp = SCOPEIT(
    XformExtTPP<T>(bc, bk, bcp, bk, XformTPP::XFORM_N2V_TPP, true),
    VNNI);
auto cpy_tpp = SCOPEIT(CpyTPP<T>(bn, bk, bk, bkp), EW_COPY);
auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(bn, bk)), EW_COPY);
auto cvt_f32_tpp = SCOPEIT((ConvertTPP<T, float>(bn, bk)), EW_COPY);
auto add_gwt_tpp = SCOPEIT((AddTPP<T, T>(bc, bk)), EW_ADD);

auto brgemm_di_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
    bn,
    bc,
    bkp,
    bn* bkp,
    nc* bc* bkp,
    0.0,
    XformTPP::XFORM_NONE_TPP,
    0,
    nk)));

#if 1
auto brgemm_dw_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
    bc,
    bk,
    bnp,
    nc* bc* bnp,
    nk* bk* bnp,
    0.0,
    XformTPP::XFORM_NONE_TPP,
    input_trans_flag,
    16)));
auto brgemm_dw_tpp_b1 = SCOPEITGEMM((BrgemmExtTPP<T, T>(
    bc,
    bk,
    bnp,
    nc* bc* bnp,
    nk* bk* bnp,
    1.0,
    XformTPP::XFORM_NONE_TPP,
    input_trans_flag,
    16)));
#else
auto brgemm_dw_tpp = SCOPEITGEMM((BrgemmExtTPP<T, float>(
    bc,
    bk,
    bnp,
    nc* bc* bnp,
    nk* bk* bnp,
    0.0,
    XformTPP::XFORM_NONE_TPP,
    input_trans_flag,
    8)));
#endif
{
  RECORD_SCOPE(gdbias, {t_grad_out});
  {
    tensor_set_zero(nk, bk, t_grad_bias);
    int threads = omp_get_max_threads();
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
            // cvt_f32_tpp(grad_out[n][k], grad_out_f32[n][k]);
            if (p > 0) {
              dropout_bwd_tpp(
                  grad_out[n][k], grad_out_f32[n][k], dp_mask[n][k]);
            }
            if (act == "relu") {
              relu_bwd_tpp(
                  grad_out_f32[n][k], grad_out_f32[n][k], relu_mask[n][k]);
            }
            grad_bias_tpp(grad_out_f32[n][k], prv_grad_bias[k]);
            cvt_tpp(grad_out_f32[n][k], grad_out[n][k]);
            n2v_tpp(grad_out[n][k], grad_out_V[n][k]);
            if (bk != bkp) {
              set_zero_col_tpp(&grad_out_K[n][k][0] + bk);
              cpy_tpp(grad_out[n][k], grad_out_K[n][k]);
            }
          }
        }
#pragma omp barrier
        omp_reduce_buf(threads, nk * bk, bias_ptrs, grad_bias[0]);
      }
    }
  }
}

{
  RECORD_SCOPE(gdi_gemm, {t_grad_out, t_wt});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int n = 0; n < nn; n++) {
      for (int c = 0; c < nc; c++) {
        if (bk != bkp)
          brgemm_di_tpp(grad_out_K[n][0], wt_TV[0][c], grad_in[n][c], nk);
        else
          brgemm_di_tpp(grad_out[n][0], wt_TV[0][c], grad_in[n][c], nk);
        if (res) {
          if (bk != bkp)
            brgemm_di_tpp(
                grad_out_K[n][0], wt_res_TV[0][c], grad_in_res[n][c], nk);
          else
            brgemm_di_tpp(
                grad_out[n][0], wt_res_TV[0][c], grad_in_res[n][c], nk);
        }
      }
    }
  }
}

#if 0
int threads = omp_get_max_threads();
auto setzero_wtptr_tpp = SCOPEIT(SetZeroTPP<float>(threads*nk*nc*bc*bk), EW_ZERO);
auto t_weight_ptrs = at::empty({threads, nk, nc, bc*bk});

constexpr int nnb=16;
{
  RECORD_SCOPE(gdw_gemm, {t_in_T, t_grad_out_V});
  {
    setzero_wtptr_tpp(t_weight_ptrs.data_ptr<float>()); 
    auto  weight_ptrs = GetVLAPtr<float>( t_weight_ptrs, { nk, nc, bc*bk});
    float *wt_ptrs[threads];
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        auto prv_grad_weight = weight_ptrs[tid];
        wt_ptrs[tid] = (float*)prv_grad_weight;
#pragma omp for collapse(3)
        for(int n=0; n<nn; n+=nnb) {
          for(int k=0; k<nk; k++) {
            for(int c=0; c<nc; c++) {
              brgemm_dw_tpp(in_T[n][c], grad_out_V[n][k], prv_grad_weight[k][c], n+nnb < nn ? nnb : nn-n);
            }
          }
        }
#pragma omp barrier
        omp_reduce_buf(threads, nk*nc*bc*bk, wt_ptrs, grad_wt_tmp[0][0]);
      }
    }
#if 1
    for(int k=0; k<nk; k++) {
      for(int c=0; c<nc; c++) {
      	n2v_wt_tpp(grad_wt_tmp[k][c], grad_wt[k][c]);
      }
    }
#endif
    if(res) {
      setzero_wtptr_tpp(t_weight_ptrs.data_ptr<float>()); 
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        auto prv_grad_weight = weight_ptrs[tid];
        wt_ptrs[tid] = (float*)prv_grad_weight;
#pragma omp for collapse(3)
        for(int n=0; n<nn; n+=nnb) {
          for(int k=0; k<nk; k++) {
            for(int c=0; c<nc; c++) {
              brgemm_dw_tpp(in_res_T[n][c], grad_out_V[n][k], prv_grad_weight[k][c], n+nnb < nn ? nnb : nn-n);
            }
          }
        }
#pragma omp barrier
        omp_reduce_buf(threads, nk*nc*bc*bk, wt_ptrs, grad_wt_res_tmp[0][0]);
      }
#if 1
      for(int k=0; k<nk; k++) {
        for(int c=0; c<nc; c++) {
      	  n2v_wt_tpp(grad_wt_res_tmp[k][c], grad_wt_res[k][c]);
        }
      }
#endif
    }
  }
}
#else
int threads = omp_get_max_threads();
std::mutex lock[nk * nc];
auto setzero_delwt_tpp = SCOPEIT(SetZeroTPP<T>(nk * nc * bc * bk), EW_ZERO);
setzero_delwt_tpp(t_grad_wt_tmp.data_ptr<T>());
if (res)
  setzero_delwt_tpp(t_grad_wt_res_tmp.data_ptr<T>());
{
  RECORD_SCOPE(gdw_gemm, {t_in_T, t_grad_out_V});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
    {
#pragma omp parallel
      {
        T tmp[bk * bc];
        int tid = omp_get_thread_num();

        int g_start = (nn * nk * nc) * tid / threads;
        int g_end = (nn * nk * nc) * (tid + 1) / threads;
        int s_ck = g_start / nn;
        int e_ck = (g_end - 1) / nn;
        int s_nn = g_start % nn;
        int e_nn = (g_end - 1) % nn;
        int start_nn, end_nn;
        for (int ck = s_ck; ck <= e_ck; ck++) {
          if (ck == s_ck)
            start_nn = s_nn;
          else
            start_nn = 0;
          if (ck == e_ck)
            end_nn = e_nn;
          else
            end_nn = nn - 1;

          int k = ck / nc;
          int c = ck % nc;
          int nnb = end_nn - start_nn + 1;

          constexpr int BS = 16;
          // auto t0 = getTime();
          // brgemm_dw_tpp(in_T[start_nn][c], grad_out_V[start_nn][k], tmp,
          // nnb);
          for (int start_nn1 = start_nn; start_nn1 <= end_nn; start_nn1 += BS) {
            if (start_nn1 == start_nn)
              brgemm_dw_tpp(
                  in_T[start_nn1][c],
                  grad_out_V[start_nn1][k],
                  tmp,
                  (start_nn1 + BS <= end_nn ? BS : end_nn - start_nn1 + 1));
            else
              brgemm_dw_tpp_b1(
                  in_T[start_nn1][c],
                  grad_out_V[start_nn1][k],
                  tmp,
                  (start_nn1 + BS <= end_nn ? BS : end_nn - start_nn1 + 1));
          }
          // auto t1 = getTime();
          // if(tid == 0) printf("T: %10.3f tid %2d: num_blk: %d, g_start %d,
          // g_end %d, s_ck %d, e_ck %d, start_nn %d, end_nn %d, k %d, c
          // %d\n",(t1-t0)*1e3, tid, (g_end-g_start), g_start, g_end, s_ck,
          // e_ck, start_nn, end_nn,k,c);
          lock[k * nc + c].lock();
          add_gwt_tpp(tmp, grad_wt_tmp[k][c], grad_wt_tmp[k][c]);
          lock[k * nc + c].unlock();
        }
      }
    }
      // printf("grad_wt_tmp %p, grad_wt %p\n",grad_wt_tmp,grad_wt);
#if 1
#pragma omp parallel for collapse(2)
    for (int k = 0; k < nk; k++) {
      for (int c = 0; c < nc; c++) {
        n2v_wt_tpp(grad_wt_tmp[k][c], grad_wt[k][c]);
      }
    }
#endif
    if (res) {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
      {
#pragma omp parallel
        {
          T tmp[bk * bc];
          int tid = omp_get_thread_num();

          int g_start = (nn * nk * nc) * tid / threads;
          int g_end = (nn * nk * nc) * (tid + 1) / threads;
          int s_ck = g_start / nn;
          int e_ck = (g_end - 1) / nn;
          int s_nn = g_start % nn;
          int e_nn = (g_end - 1) % nn;
          int start_nn, end_nn;
          for (int ck = s_ck; ck <= e_ck; ck++) {
            if (ck == s_ck)
              start_nn = s_nn;
            else
              start_nn = 0;
            if (ck == e_ck)
              end_nn = e_nn;
            else
              end_nn = nn - 1;

            int k = ck / nc;
            int c = ck % nc;
            int nnb = end_nn - start_nn + 1;
            constexpr int BS = 16;
            // auto t0 = getTime();
            // brgemm_dw_tpp(in_res_T[start_nn][c], grad_out_V[start_nn][k],
            // tmp, nnb);
            for (int start_nn1 = start_nn; start_nn1 <= end_nn;
                 start_nn1 += BS) {
              if (start_nn1 == start_nn)
                brgemm_dw_tpp(
                    in_res_T[start_nn1][c],
                    grad_out_V[start_nn1][k],
                    tmp,
                    (start_nn1 + BS <= end_nn ? BS : end_nn - start_nn1 + 1));
              else
                brgemm_dw_tpp_b1(
                    in_res_T[start_nn1][c],
                    grad_out_V[start_nn1][k],
                    tmp,
                    (start_nn1 + BS <= end_nn ? BS : end_nn - start_nn1 + 1));
            }
            // auto t1 = getTime();
            lock[k * nc + c].lock();
            add_gwt_tpp(tmp, grad_wt_res_tmp[k][c], grad_wt_res_tmp[k][c]);
            lock[k * nc + c].unlock();
          }
        }
      }
#pragma omp parallel for collapse(2)
      for (int k = 0; k < nk; k++) {
        for (int c = 0; c < nc; c++) {
          n2v_wt_tpp(grad_wt_res_tmp[k][c], grad_wt_res[k][c]);
        }
      }
    }
  }
}
#endif
#if 0
{
  RECORD_SCOPE(gdw_gemm, {t_in_T, t_grad_out_V});
  {
    tensor_set_zero(nk*bk, nc*bc, t_grad_wt);
    if(res)
      tensor_set_zero(nk*bk, nc*bc, t_grad_wt_res);

    t_grad_out_V = t_grad_out_V.contiguous();
    t_grad_wt = t_grad_wt.contiguous();
    if(res)
      t_grad_wt_res = t_grad_wt_res.contiguous();

    constexpr int nnb = 32;
    for(int n=0; n<nn; n+=nnb) {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for(int k=0; k<nk; k++) {
        for(int c=0; c<nc; c++) {
          brgemm_dw_tpp(in_T[n][c], grad_out_V[n][k], grad_wt[k][c], n+nnb < nn ? nnb : nn-n);
          if(res) {
            brgemm_dw_tpp(in_res_T[n][c], grad_out_V[n][k], grad_wt_res[k][c], n+nnb < nn ? nnb : nn-n);
          }
        }
      }
    }
  }
}
#endif
if (res) {
  return {t_grad_in, t_grad_in_res, t_grad_wt, t_grad_wt_res, t_grad_bias};
} else {
  return {t_grad_in, t_grad_wt, t_grad_bias};
}

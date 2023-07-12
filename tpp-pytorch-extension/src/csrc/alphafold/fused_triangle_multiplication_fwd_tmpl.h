/******************************************************************************
 * Copyright (c) 2023 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Narendra Chaudhary (Intel Corp.)
 ******************************************************************************/

RECORD_FUNCTION(
    "Triangle multiplication forward",
    std::vector<c10::IValue>({act, mask})); // For recording time

/* act shape --> [764, 764, 64] */
/* mask_shape ---> [764, 764, 1] */
/* left_projection_weight -> [32, 64], left_projection_bias -> [32] */
/* right_projection_weight -> [32, 64], right_projection_bias -> [32] */
/* left_gate_weight -> [32, 64], left_gate_weight_bias -> [32] */
/* right_gate_weight -> [32, 64], right_gate_weight_bias -> [32] */

int64_t Bp_t = act.size(0);
int64_t Sp_t = act.size(1);
int64_t act_dim = act.size(2);

int64_t num_intermediate_channel = left_projection_weight.size(0);

// act = at::layer_norm(act, act_dim, layer_norm_input_weight,
// layer_norm_input_bias).contiguous();

int64_t S_t = Sp_t;
if (Sp_t % TRI_BLOCKSIZE != 0) {
  S_t = (Sp_t / TRI_BLOCKSIZE + 1) * TRI_BLOCKSIZE;
  auto act_pad = act.new_zeros({Bp_t, S_t - Sp_t, act_dim});
  auto mask_pad = mask.new_zeros({Bp_t, S_t - Sp_t, 1});

  act = at::cat({act, act_pad}, 1);
  mask = at::cat({mask, mask_pad}, 1);
}

int64_t B_t = Bp_t;
if (Bp_t % TRI_BLOCKSIZE != 0) {
  B_t = (Bp_t / TRI_BLOCKSIZE + 1) * TRI_BLOCKSIZE;
  auto act_pad = act.new_zeros({B_t - Bp_t, S_t, act_dim});
  auto mask_pad = mask.new_zeros({B_t - Bp_t, S_t, 1});

  act = at::cat({act, act_pad}, 0);
  mask = at::cat({mask, mask_pad}, 0);
}

auto act_a = GetVLAPtr<T>(act, {S_t, act_dim});

auto layernorm =
    SCOPEIT(LayerNormFwdTPP<T>(1, TRI_BLOCKSIZE, act_dim, 0.00001), LAYER_NORM);
auto input_gamma_a = GetVLAPtr<T>(layer_norm_input_weight, {1L});
auto input_beta_a = GetVLAPtr<T>(layer_norm_input_bias, {1L});

{
  RECORD_SCOPE(layer_norm_input, {act, layer_norm_input_weight});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int i = 0; i < B_t; i++) {
      for (int j = 0; j < S_t; j += TRI_BLOCKSIZE) {
        T tmp_mean[act_dim];
        T tmp_var[act_dim];
        layernorm(
            &act_a[i][j][0],
            &input_gamma_a[0][0],
            &input_beta_a[0][0],
            &tmp_mean[0],
            &tmp_var[0],
            &act_a[i][j][0]);
      }
    }
  }
}

auto input_act = act;

auto mask_a = GetVLAPtr<T>(mask, {S_t, 1L});
auto input_act_a = GetVLAPtr<T>(input_act, {S_t, act_dim});

int64_t first_block_dim = B_t / TRI_BLOCKSIZE;
int64_t second_block_dim = S_t / TRI_BLOCKSIZE;
if (equation_flag != 0) {
  first_block_dim = S_t / TRI_BLOCKSIZE;
  second_block_dim = B_t / TRI_BLOCKSIZE;
}

//  Generate intermediate tensors in a blocked formet example
//  [num_intermediate_channel][B_t/TRI_BLOCKSIZE][S_t/TRI_BLOCKSIZE][TRI_BLOCKSIZE][TRI_BLOCKSIZE]
auto left_proj_act = act.new_empty({num_intermediate_channel,
                                    first_block_dim,
                                    second_block_dim,
                                    TRI_BLOCKSIZE,
                                    TRI_BLOCKSIZE});
auto left_proj_act_a = GetVLAPtr<T>(
    left_proj_act,
    {first_block_dim, second_block_dim, TRI_BLOCKSIZE, TRI_BLOCKSIZE});
auto right_proj_act = act.new_empty({num_intermediate_channel,
                                     first_block_dim,
                                     second_block_dim,
                                     TRI_BLOCKSIZE,
                                     TRI_BLOCKSIZE});
auto right_proj_act_a = GetVLAPtr<T>(
    right_proj_act,
    {first_block_dim, second_block_dim, TRI_BLOCKSIZE, TRI_BLOCKSIZE});

auto left_projection_weight_a = GetVLAPtr<T>(left_projection_weight, {act_dim});
auto left_projection_bias_a = GetVLAPtr<T>(left_projection_bias, {1L});
auto right_projection_weight_a =
    GetVLAPtr<T>(right_projection_weight, {act_dim});
auto right_projection_bias_a = GetVLAPtr<T>(right_projection_bias, {1L});

// ---------- Transpose ------------
auto left_trans_proj_weight =
    left_projection_weight.new_empty({act_dim, num_intermediate_channel});
auto left_trans_proj_weight_a =
    GetVLAPtr<T>(left_trans_proj_weight, {num_intermediate_channel});

auto left_trans_gate_weight =
    left_projection_weight.new_empty({act_dim, num_intermediate_channel});
auto left_trans_gate_weight_a =
    GetVLAPtr<T>(left_trans_gate_weight, {num_intermediate_channel});

auto right_trans_proj_weight =
    left_projection_weight.new_empty({act_dim, num_intermediate_channel});
auto right_trans_proj_weight_a =
    GetVLAPtr<T>(right_trans_proj_weight, {num_intermediate_channel});

auto right_trans_gate_weight =
    left_projection_weight.new_empty({act_dim, num_intermediate_channel});
auto right_trans_gate_weight_a =
    GetVLAPtr<T>(right_trans_gate_weight, {num_intermediate_channel});

auto output_trans_proj_weight =
    left_projection_weight.new_empty({act_dim, act_dim});
auto output_trans_proj_weight_a =
    GetVLAPtr<T>(output_trans_proj_weight, {act_dim});

auto gating_linear_trans_weight =
    left_projection_weight.new_empty({act_dim, act_dim});
auto gating_linear_trans_weight_a =
    GetVLAPtr<T>(gating_linear_trans_weight, {act_dim});

// ---------------------------------

auto left_gate_weight_a = GetVLAPtr<T>(left_gate_weight, {act_dim});
auto left_gate_bias_a = GetVLAPtr<T>(left_gate_bias, {1L});

auto right_gate_weight_a = GetVLAPtr<T>(right_gate_weight, {act_dim});
auto right_gate_bias_a = GetVLAPtr<T>(right_gate_bias, {1L});

auto output_projection_weight_a =
    GetVLAPtr<T>(output_projection_weight, {act_dim});
auto output_projection_bias_a = GetVLAPtr<T>(output_projection_bias, {1L});

auto gating_linear_weight_a = GetVLAPtr<T>(gating_linear_weight, {act_dim});
auto gating_linear_bias_a = GetVLAPtr<T>(gating_linear_bias, {1L});

int64_t lda = act_dim;
int64_t ldb = num_intermediate_channel;
int64_t ldc = num_intermediate_channel;
auto proj_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
    TRI_BLOCKSIZE,
    num_intermediate_channel,
    act_dim,
    0,
    0,
    lda,
    ldb,
    ldc,
    0.0,
    0,
    1)));

auto weight_trans_tpp = SCOPEIT(
    XformExtTPP<T>(
        num_intermediate_channel,
        act_dim,
        XformTPP::XFORM_XPOSE_TPP),
    XPOSE);

auto proj_trans_tpp = SCOPEIT(
    XformExtTPP<T>(
        TRI_BLOCKSIZE,
        num_intermediate_channel,
        num_intermediate_channel,
        TRI_BLOCKSIZE,
        num_intermediate_channel,
        B_t* S_t,
        XformTPP::XFORM_XPOSE_TPP),
    XPOSE);

auto add_bias_tpp =
    SCOPEIT(AddBiasTPP<T>(TRI_BLOCKSIZE, num_intermediate_channel, ldc), BIAS);

auto mul_broad_tpp = SCOPEIT(
    (BCastMulTPP<T, T>(TRI_BLOCKSIZE, num_intermediate_channel)),
    EW_MUL);
auto mul_tpp =
    SCOPEIT((MulTPP<T, T>(TRI_BLOCKSIZE, num_intermediate_channel)), EW_MUL);

auto sigmoid_tpp = SCOPEIT(
    SiLUFwdTPP<T>(
        TRI_BLOCKSIZE,
        num_intermediate_channel,
        num_intermediate_channel,
        num_intermediate_channel),
    EW_SCL);

{
  RECORD_SCOPE(proj_gemm, {act, mask, left_projection_weight});
  {
    weight_trans_tpp(
        &left_projection_weight_a[0][0],
        &left_trans_proj_weight_a[0][0]); // Transpose weights for the linear
                                          // layers
    weight_trans_tpp(
        &left_gate_weight_a[0][0], &left_trans_gate_weight_a[0][0]);

    weight_trans_tpp(
        &right_projection_weight_a[0][0], &right_trans_proj_weight_a[0][0]);
    weight_trans_tpp(
        &right_gate_weight_a[0][0], &right_trans_gate_weight_a[0][0]);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
// for (int i = 0; i < B_t; i++) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < B_t; i += TRI_BLOCKSIZE) {
      for (int j = 0; j < S_t; j += TRI_BLOCKSIZE) {
        T tmp[TRI_BLOCKSIZE][num_intermediate_channel];
        T tmp_gate_values[TRI_BLOCKSIZE][num_intermediate_channel];
        T tmp_proj[TRI_BLOCKSIZE][num_intermediate_channel];

        for (int ib = 0; ib < TRI_BLOCKSIZE; ib++) {
          proj_brgemm_tpp(
              &act_a[i + ib][j][0],
              &left_trans_proj_weight_a[0][0],
              &tmp[0][0],
              1);
          add_bias_tpp(&left_projection_bias_a[0][0], &tmp[0][0]);
          mul_broad_tpp(&mask_a[i + ib][j][0], &tmp[0][0], &tmp_proj[0][0]);

          proj_brgemm_tpp(
              &act_a[i + ib][j][0],
              &left_trans_gate_weight_a[0][0],
              &tmp[0][0],
              1);
          add_bias_tpp(&left_gate_bias_a[0][0], &tmp[0][0]);
          sigmoid_tpp(&tmp[0][0], &tmp[0][0], &tmp_gate_values[0][0]);
          mul_tpp(&tmp_proj[0][0], &tmp_gate_values[0][0], &tmp_proj[0][0]);

          if (equation_flag == 0)
            proj_trans_tpp(
                &tmp_proj[0][0],
                &left_proj_act_a[0][i / TRI_BLOCKSIZE][j / TRI_BLOCKSIZE][ib]
                                [0]);
          else
            proj_trans_tpp(
                &tmp_proj[0][0],
                &left_proj_act_a[0][j / TRI_BLOCKSIZE][i / TRI_BLOCKSIZE][ib]
                                [0]);

          proj_brgemm_tpp(
              &act_a[i + ib][j][0],
              &right_trans_proj_weight_a[0][0],
              &tmp[0][0],
              1);
          add_bias_tpp(&right_projection_bias_a[0][0], &tmp[0][0]);
          mul_broad_tpp(&mask_a[i + ib][j][0], &tmp[0][0], &tmp_proj[0][0]);

          proj_brgemm_tpp(
              &act_a[i + ib][j][0],
              &right_trans_gate_weight_a[0][0],
              &tmp[0][0],
              1);
          add_bias_tpp(&right_gate_bias_a[0][0], &tmp[0][0]);
          sigmoid_tpp(&tmp[0][0], &tmp[0][0], &tmp_gate_values[0][0]);
          mul_tpp(&tmp_proj[0][0], &tmp_gate_values[0][0], &tmp_proj[0][0]);

          if (equation_flag == 0)
            proj_trans_tpp(
                &tmp_proj[0][0],
                &right_proj_act_a[0][i / TRI_BLOCKSIZE][j / TRI_BLOCKSIZE][ib]
                                 [0]);
          else
            proj_trans_tpp(
                &tmp_proj[0][0],
                &right_proj_act_a[0][j / TRI_BLOCKSIZE][i / TRI_BLOCKSIZE][ib]
                                 [0]);
        }
      }
    }
  }
}

if (equation_flag == 0) { // "Outgoing" edges equation = 'ikc,jkc->ijc'
  // act = at::einsum("ikc,jkc->ijc", {left_proj_act, right_proj_act}); // [764,
  // 764, 64], [764, 764, 64] act =
  // at::permute(at::bmm(at::permute(left_proj_act, {2, 0, 1}),
  // at::permute(right_proj_act, {2, 1, 0})), {1, 2, 0}); left_proj_act =
  // at::permute(left_proj_act, {2, 0, 1}).contiguous();             // [B_t,
  // S_t, 64]  --> [64, B_t, S_t] right_proj_act = at::permute(right_proj_act,
  // {2, 1, 0}).contiguous();           // [B_t, S_t, 64]  --> [64, S_t, B_t]
  // // act = at::bmm(left_proj_act, right_proj_act); // [64, 764, 764], [64,
  // 764, 764]

  lda = TRI_BLOCKSIZE;
  ldb = TRI_BLOCKSIZE;
  ldc = TRI_BLOCKSIZE;
  int64_t str_a = TRI_BLOCKSIZE * TRI_BLOCKSIZE;
  int64_t str_b = TRI_BLOCKSIZE * TRI_BLOCKSIZE;
  auto equation_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      TRI_BLOCKSIZE,
      TRI_BLOCKSIZE,
      TRI_BLOCKSIZE,
      str_a,
      str_b,
      lda,
      ldb,
      ldc,
      0.0,
      0,
      1)));
  // T>(TRI_BLOCKSIZE, TRI_BLOCKSIZE, TRI_BLOCKSIZE, 0, 0, lda, ldb, ldc, 1.0,
  // 0, 1)));

  auto equation_trans_tpp = SCOPEIT(
      XformExtTPP<T>(TRI_BLOCKSIZE, TRI_BLOCKSIZE, XformTPP::XFORM_XPOSE_TPP),
      XPOSE);

  act = act.new_empty({num_intermediate_channel,
                       B_t / TRI_BLOCKSIZE,
                       B_t / TRI_BLOCKSIZE,
                       TRI_BLOCKSIZE,
                       TRI_BLOCKSIZE});
  auto act_an = GetVLAPtr<T>(
      act,
      {B_t / TRI_BLOCKSIZE, B_t / TRI_BLOCKSIZE, TRI_BLOCKSIZE, TRI_BLOCKSIZE});

  {
    RECORD_SCOPE(eq_bmm, {left_proj_act, right_proj_act});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
      for (int a = 0; a < num_intermediate_channel; a++) {
        for (int i = 0; i < B_t; i += TRI_BLOCKSIZE) {
          for (int j = 0; j < B_t; j += TRI_BLOCKSIZE) {
            // equation_brgemm_tpp(&left_proj_act_a[a][i][0],
            // &right_proj_act_a[a][0][j], &act_a[a][i][j], S_t/TRI_BLOCKSIZE);
            T tmp[S_t / TRI_BLOCKSIZE][TRI_BLOCKSIZE][TRI_BLOCKSIZE];
            for (int ib = 0; ib < S_t / TRI_BLOCKSIZE; ib++)
              equation_trans_tpp(
                  &right_proj_act_a[a][j / TRI_BLOCKSIZE][ib][0][0],
                  &tmp[ib][0][0]);

            equation_brgemm_tpp(
                &left_proj_act_a[a][i / TRI_BLOCKSIZE][0][0][0],
                &tmp[0][0][0],
                &act_an[a][i / TRI_BLOCKSIZE][j / TRI_BLOCKSIZE][0][0],
                S_t / TRI_BLOCKSIZE);

            // T tmp[TRI_BLOCKSIZE][TRI_BLOCKSIZE];
            // for(int ib = 0; ib < S_t/TRI_BLOCKSIZE ; ib++){
            //     equation_trans_tpp(&right_proj_act_a[a][j/TRI_BLOCKSIZE][ib][0][0],
            //     &tmp[0][0]);

            //     equation_brgemm_tpp(&left_proj_act_a[a][i/TRI_BLOCKSIZE][ib][0][0],
            //     &tmp[0][0],
            //                 &act_an[a][i/TRI_BLOCKSIZE][j/TRI_BLOCKSIZE][0][0],
            //                 1);
            // }
          }
        }
      }
    }
  }
  // act = at::permute(act, {1, 2, 0});
  act = act.permute({1, 3, 2, 4, 0})
            .reshape({B_t, B_t, num_intermediate_channel});
} else { // "Incoming" edges equation: 'kjc,kic->ijc'
  // act = at::einsum("kjc,kic->ijc", {left_proj_act, right_proj_act});

  lda = TRI_BLOCKSIZE;
  ldb = TRI_BLOCKSIZE;
  ldc = TRI_BLOCKSIZE;
  int64_t str_a = TRI_BLOCKSIZE * TRI_BLOCKSIZE;
  int64_t str_b = TRI_BLOCKSIZE * TRI_BLOCKSIZE;
  auto equation_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
      TRI_BLOCKSIZE,
      TRI_BLOCKSIZE,
      TRI_BLOCKSIZE,
      str_a,
      str_b,
      lda,
      ldb,
      ldc,
      0.0,
      1,
      1)));
  // T>(TRI_BLOCKSIZE, TRI_BLOCKSIZE, TRI_BLOCKSIZE, 0, 0, lda, ldb, ldc, 1.0,
  // 1, 1)));

  act = act.new_empty({num_intermediate_channel,
                       S_t / TRI_BLOCKSIZE,
                       S_t / TRI_BLOCKSIZE,
                       TRI_BLOCKSIZE,
                       TRI_BLOCKSIZE});
  auto act_an = GetVLAPtr<T>(
      act,
      {S_t / TRI_BLOCKSIZE, S_t / TRI_BLOCKSIZE, TRI_BLOCKSIZE, TRI_BLOCKSIZE});

  {
    RECORD_SCOPE(eq_bmm, {left_proj_act, right_proj_act});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
      for (int a = 0; a < num_intermediate_channel; a++) {
        for (int i = 0; i < S_t; i += TRI_BLOCKSIZE) {
          for (int j = 0; j < S_t; j += TRI_BLOCKSIZE) {
            // equation_brgemm_tpp(&left_proj_act_a[a][i][0],
            // &right_proj_act_a[a][0][j], &act_a[a][i][j], B_t/TRI_BLOCKSIZE);
            equation_brgemm_tpp(
                &left_proj_act_a[a][i / TRI_BLOCKSIZE][0][0][0],
                &right_proj_act_a[a][j / TRI_BLOCKSIZE][0][0][0],
                &act_an[a][i / TRI_BLOCKSIZE][j / TRI_BLOCKSIZE][0][0],
                B_t / TRI_BLOCKSIZE);
            // for(int ib = 0; ib < B_t/TRI_BLOCKSIZE ; ib++){
            //     equation_brgemm_tpp(&left_proj_act_a[a][i/TRI_BLOCKSIZE][ib][0][0],
            //     &right_proj_act_a[a][j/TRI_BLOCKSIZE][ib][0][0],
            //             &act_an[a][i/TRI_BLOCKSIZE][j/TRI_BLOCKSIZE][0][0],
            //             1);
            // }
          }
        }
      }
    }
  }
  // act = at::permute(act, {2, 1, 0});
  act = act.permute({2, 4, 1, 3, 0})
            .reshape({S_t, S_t, num_intermediate_channel});
}

// act = at::layer_norm(act, act_dim, center_layer_norm_weight,
// center_layer_norm_bias);
auto center_gamma_a = GetVLAPtr<T>(center_layer_norm_weight, {1L});
auto center_beta_a = GetVLAPtr<T>(center_layer_norm_bias, {1L});

lda = act_dim;
ldb = act_dim;
ldc = act_dim;
auto outgate_brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        T>(TRI_BLOCKSIZE, act_dim, act_dim, 0, 0, lda, ldb, ldc, 0.0, 0, 1)));

auto outgate_weight_trans_tpp =
    SCOPEIT(XformExtTPP<T>(act_dim, act_dim, XformTPP::XFORM_XPOSE_TPP), XPOSE);

auto outgate_add_bias_tpp =
    SCOPEIT(AddBiasTPP<T>(TRI_BLOCKSIZE, act_dim, ldc), BIAS);

auto outgate_copy_tpp = SCOPEIT(CpyTPP<T>(TRI_BLOCKSIZE, act_dim), EW_COPY);

auto outgate_mul_tpp = SCOPEIT((MulTPP<T, T>(TRI_BLOCKSIZE, act_dim)), EW_MUL);
auto outgate_sigmoid_tpp =
    SCOPEIT(SiLUFwdTPP<T>(TRI_BLOCKSIZE, act_dim, act_dim, act_dim), EW_SCL);

// act = at::add(at::einsum("bqa,ca->bqc", {act, output_projection_weight}),
// output_projection_bias).contiguous(); act = at::mul(act,
// at::sigmoid(at::add(at::einsum("bqa,ca->bqc", {input_act,
// gating_linear_weight}), gating_linear_bias)));

act_a = GetVLAPtr<T>(act, {S_t, act_dim});
{
  RECORD_SCOPE(out_gemm, {act, output_projection_weight});
  {
    outgate_weight_trans_tpp(
        &output_projection_weight_a[0][0], &output_trans_proj_weight_a[0][0]);

    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int i = 0; i < B_t; i++) {
      for (int j = 0; j < S_t; j += TRI_BLOCKSIZE) {
        T tmp[TRI_BLOCKSIZE][act_dim];
        T tmp_mean[act_dim];
        T tmp_var[act_dim];

        layernorm(
            &act_a[i][j][0],
            &center_gamma_a[0][0],
            &center_beta_a[0][0],
            &tmp_mean[0],
            &tmp_var[0],
            &act_a[i][j][0]);

        outgate_brgemm_tpp(
            &act_a[i][j][0], &output_trans_proj_weight_a[0][0], &tmp[0][0], 1);
        outgate_add_bias_tpp(&output_projection_bias_a[0][0], &tmp[0][0]);
        outgate_copy_tpp(&tmp[0][0], &act_a[i][j][0]);
      }
    }
  }
}

act_a = GetVLAPtr<T>(act, {S_t, act_dim});
{
  RECORD_SCOPE(gate_gemm, {act, gating_linear_weight});
  {
    outgate_weight_trans_tpp(
        &gating_linear_weight_a[0][0], &gating_linear_trans_weight_a[0][0]);

    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int i = 0; i < B_t; i++) {
      for (int j = 0; j < S_t; j += TRI_BLOCKSIZE) {
        T tmp[TRI_BLOCKSIZE][act_dim];
        T tmp_gate_values[TRI_BLOCKSIZE][act_dim];

        outgate_brgemm_tpp(
            &input_act_a[i][j][0],
            &gating_linear_trans_weight_a[0][0],
            &tmp[0][0],
            1);
        outgate_add_bias_tpp(&gating_linear_bias_a[0][0], &tmp[0][0]);
        outgate_sigmoid_tpp(&tmp[0][0], &tmp[0][0], &tmp_gate_values[0][0]);
        outgate_mul_tpp(
            &act_a[i][j][0], &tmp_gate_values[0][0], &act_a[i][j][0]);
      }
    }
  }
}

if (S_t != Sp_t) {
  act = act.narrow(1, 0, Sp_t);
}

if (B_t != Bp_t) {
  act = act.narrow(0, 0, Bp_t);
}

return act;
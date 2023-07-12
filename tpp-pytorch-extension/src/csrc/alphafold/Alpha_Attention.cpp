/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Narendra Chaudhary (Intel Corp.)
 ******************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <torch/extension.h>
#include <cmath>
#include <iostream>
#include <tuple>

#include <ATen/record_function.h>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
#include "tensor_helper.h"

#define TRI_BLOCKSIZE 32

#define QKV_BLOCKSIZE 64
#define A_BLOCKSIZE 64
#define C_BLOCKSIZE 64

REGISTER_SCOPE(alpha_q_gemm, "alpha_q_gemm");
REGISTER_SCOPE(alpha_k_gemm, "alpha_k_gemm");
REGISTER_SCOPE(alpha_v_gemm, "alpha_v_gemm");

REGISTER_SCOPE(alpha_a_gemm, "alpha_a_gemm");
REGISTER_SCOPE(alpha_c_gemm, "alpha_c_gemm");

REGISTER_SCOPE(proj_gemm, "proj_gemm");
REGISTER_SCOPE(out_gemm, "out_gemm");
REGISTER_SCOPE(gate_gemm, "gate_gemm");
REGISTER_SCOPE(eq_bmm, "eq_gemm");
REGISTER_SCOPE(layer_norm_input, "layer_norm_input");

at::Tensor fused_gating_attention_fwd(
    at::Tensor& q_data,
    at::Tensor& m_data,
    at::Tensor& bias,
    at::Tensor& nonbatched_bias,
    at::Tensor& query_w,
    at::Tensor& key_w,
    at::Tensor& value_w,
    at::Tensor& gating_w,
    at::Tensor& gating_b,
    at::Tensor& output_w,
    at::Tensor& output_b,
    int key_dim,
    int value_dim) {
  GlobalPass _gp(FWD);
  if (q_data.dtype() == at::kFloat) {
    typedef float T;
#include "fused_gating_attention_fwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "fused_gating_attention_fwd_tmpl_bf16.h"
  }
}

at::Tensor fused_traingle_multiplication_fwd(
    at::Tensor& act,
    at::Tensor& mask,
    int equation_flag,
    at::Tensor& layer_norm_input_weight,
    at::Tensor& layer_norm_input_bias,
    at::Tensor& left_projection_weight,
    at::Tensor& left_projection_bias,
    at::Tensor& right_projection_weight,
    at::Tensor& right_projection_bias,
    at::Tensor& left_gate_weight,
    at::Tensor& left_gate_bias,
    at::Tensor& right_gate_weight,
    at::Tensor& right_gate_bias,
    at::Tensor& center_layer_norm_weight,
    at::Tensor& center_layer_norm_bias,
    at::Tensor& output_projection_weight,
    at::Tensor& output_projection_bias,
    at::Tensor& gating_linear_weight,
    at::Tensor& gating_linear_bias) {
  GlobalPass _gp(FWD);
  if (act.dtype() == at::kFloat) {
    typedef float T;
#include "fused_triangle_multiplication_fwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "fused_triangle_multiplication_fwd_tmpl_bf16.h"
  }
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
REGISTER_SUBMODULE(_alpha_attention, m) {
  m.def("forward", &fused_gating_attention_fwd, "Gating attention forward");
  m.def(
      "trianglemulti_forward",
      &fused_traingle_multiplication_fwd,
      "Traingle Multiplication forward");
}

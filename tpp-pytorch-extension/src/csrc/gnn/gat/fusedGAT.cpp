/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)
 ******************************************************************************/

#include <ATen/record_function.h>
#include <torch/extension.h>
#include <cstdlib>

#include <iostream>
#include <mutex>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
#include "tensor_helper.h"

static int my_rank = guess_mpi_rank();

REGISTER_SCOPE(gao_gemm, "gao_gemm");
REGISTER_SCOPE(gadi_gemm, "gadi_gemm");
REGISTER_SCOPE(gadw_gemm, "gadw_gemm");
REGISTER_SCOPE(gobias, "gobias");
REGISTER_SCOPE(gadbias, "gadbias");
REGISTER_SCOPE(gao_dropout, "gao_dropout");
REGISTER_SCOPE(gado_dropout, "gado_dropout");
REGISTER_SCOPE(go_lrelu, "go_lrelu");
REGISTER_SCOPE(gdo_lrelu, "gdo_lrelu");
REGISTER_SCOPE(go_attn, "go_attn");
REGISTER_SCOPE(gdo_attn, "gdo_attn");
REGISTER_SCOPE(go_mlp_attn, "go_mlp_attn");
REGISTER_SCOPE(gdo_mlp_attn, "gdo_mlp_attn");
REGISTER_SCOPE(ga_dattn_dbias_din, "ga_dattn_dbias_din");

// ######################################## FUSED GAT MLP & ATTENTION
// ################################################

std::vector<at::Tensor> fused_gat_mlp_attn_fwd(
    long align,
    std::string act,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_gat_mlp_attn_flat_fwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_gat_mlp_attn_flat_fwd.h"
  }
}

std::vector<at::Tensor> fused_gat_mlp_attn_bwd(
    long align,
    std::string act,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_gat_mlp_attn_flat_bwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_gat_mlp_attn_flat_bwd.h"
  }
}

// ######################################## Dropout
// ################################################

std::vector<at::Tensor> gat_dropout_fwd(
    float p,
    at::Tensor inp,
    bool training) {
  GlobalPass _gp(FWD);
  if (inp.dtype() == at::kFloat) {
    typedef float T;
#include "dropout_fwd.h"
  } else {
    typedef bfloat16 T;
#include "dropout_fwd.h"
  }
}

at::Tensor gat_dropout_bwd(float p, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "dropout_bwd.h"
  } else {
    typedef bfloat16 T;
#include "dropout_bwd.h"
  }
}

// ######################################## Leaky ReLU
// ################################################

std::vector<at::Tensor> leakyrelu_fwd(float alpha, at::Tensor inp) {
  GlobalPass _gp(FWD);
  if (inp.dtype() == at::kFloat) {
    // std::cout << " In Cpp "<< inp << std::endl;
    typedef float T;
#include "leakyrelu_fwd.h"
  } else {
    typedef bfloat16 T;
#include "leakyrelu_fwd.h"
  }
}

at::Tensor leakyrelu_bwd(float alpha, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "leakyrelu_bwd.h"
  } else {
    typedef bfloat16 T;
#include "leakyrelu_bwd.h"
  }
}

REGISTER_SUBMODULE(_fused_gsage, m) {
  m.def(
      "fused_gat_mlp_attn_fwd",
      &fused_gat_mlp_attn_fwd,
      "Tpp GAT fused MLP-Attention forward");
  m.def(
      "fused_gat_mlp_attn_bwd",
      &fused_gat_mlp_attn_bwd,
      "Tpp GAT fused MLP-Attention backward");
  m.def("gat_dropout_fwd", &gat_dropout_fwd, "Tpp Optimized Dropout FWD");
  m.def("gat_dropout_bwd", &gat_dropout_bwd, "Tpp Optimized Dropout BWD");
  m.def("leakyrelu_fwd", &leakyrelu_fwd, "Tpp Optimized Leaky Relu FWD");
  m.def("leakyrelu_bwd", &leakyrelu_bwd, "Tpp Optimized Leaky Relu BWD");
}

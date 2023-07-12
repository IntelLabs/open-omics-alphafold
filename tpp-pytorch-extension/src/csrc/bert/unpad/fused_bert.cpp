/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#include <ATen/record_function.h>
// #include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#ifndef NO_PARLOOPER
#include "threaded_loops.h"
#endif
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
#include "tensor_helper.h"

static int my_rank = guess_mpi_rank();
static int large_cache_opt = true;

REGISTER_LOCAL_SCOPE(b_emb, "b_emb");
REGISTER_LOCAL_SCOPE(q_gemm, "q_gemm");
REGISTER_LOCAL_SCOPE(k_gemm, "k_gemm");
REGISTER_LOCAL_SCOPE(v_gemm, "v_gemm");
REGISTER_LOCAL_SCOPE(ac_gemm, "ac_gemm");
REGISTER_LOCAL_SCOPE(o_gemm, "o_gemm");
REGISTER_LOCAL_SCOPE(i_gemm, "i_gemm");

REGISTER_LOCAL_SCOPE(db_emb, "db_emb");
REGISTER_LOCAL_SCOPE(diq_gemm, "diq_gemm");
REGISTER_LOCAL_SCOPE(dik_gemm, "dik_gemm");
REGISTER_LOCAL_SCOPE(div_gemm, "div_gemm");
REGISTER_LOCAL_SCOPE(dica_gemm, "dica_gemm");
REGISTER_LOCAL_SCOPE(dii_gemm, "dii_gemm");
REGISTER_LOCAL_SCOPE(dio_gemm, "dio_gemm");
REGISTER_LOCAL_SCOPE(dwqkv_gemm, "dwqkv_gemm");
REGISTER_LOCAL_SCOPE(dwq_gemm, "dwq_gemm");
REGISTER_LOCAL_SCOPE(dwk_gemm, "dwk_gemm");
REGISTER_LOCAL_SCOPE(dwv_gemm, "dwv_gemm");
REGISTER_LOCAL_SCOPE(dwa_gemm, "dwa_gemm");
REGISTER_LOCAL_SCOPE(dwc_gemm, "dwc_gemm");
REGISTER_LOCAL_SCOPE(dac_gemm, "dac_gemm");
REGISTER_LOCAL_SCOPE(dwi_gemm, "dwi_gemm");
REGISTER_LOCAL_SCOPE(dwo_gemm, "dwo_gemm");
REGISTER_LOCAL_SCOPE(dqkv_bias, "dqkv_bias");
REGISTER_LOCAL_SCOPE(di_bias, "di_bias");
REGISTER_LOCAL_SCOPE(do_bias, "do_bias");

static std::vector<at::Tensor> fused_self_attention_fwd(
    float p,
    std::vector<at::Tensor> inputs,
    bool training) {
  GlobalPass _gp(FWD);
  if (inputs[6].dtype() == at::kFloat) {
    typedef float T;
#include "fused_self_attention_fwd_tmpl.h"
  } else if (inputs[6].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "fused_self_attention_fwd_tmpl.h"
  } else if (inputs[6].dtype() == at::kBFloat8) {
    typedef bfloat8 T;
#include "fused_self_attention_fwd_tmpl.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

static std::vector<at::Tensor> fused_self_attention_bwd(
    float p,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_self_attention_bwd_tmpl.h"
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "fused_self_attention_bwd_tmpl.h"
  } else if (inputs[0].dtype() == at::kBFloat8) {
    typedef bfloat8 T;
#include "fused_self_attention_bwd_tmpl.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

static std::vector<at::Tensor> fused_dense_dropout_layernorm_fwd(
    float p,
    float eps,
    std::vector<at::Tensor> inputs,
    bool training) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat && inputs[4].dtype() == at::kFloat) {
    typedef float T;
    typedef float LT;
#include "fused_dense_dropout_layernorm_fwd_tmpl.h"
  } else if (
      inputs[0].dtype() == at::kBFloat16 && inputs[4].dtype() == at::kFloat) {
    typedef bfloat16 T;
    typedef float LT;
#include "fused_dense_dropout_layernorm_fwd_tmpl.h"
  } else if (
      inputs[0].dtype() == at::kBFloat8 && inputs[4].dtype() == at::kFloat) {
    typedef bfloat8 T;
    typedef float LT;
#include "fused_dense_dropout_layernorm_fwd_tmpl.h"
  } else if (
      inputs[0].dtype() == at::kBFloat16 &&
      inputs[4].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
    typedef bfloat16 LT;
#include "fused_dense_dropout_layernorm_fwd_tmpl.h"
  } else if (
      inputs[0].dtype() == at::kBFloat8 && inputs[4].dtype() == at::kBFloat8) {
    typedef bfloat8 T;
    typedef bfloat8 LT;
#include "fused_dense_dropout_layernorm_fwd_tmpl.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

static std::vector<at::Tensor> fused_dense_dropout_layernorm_bwd(
    float p,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat && inputs[3].dtype() == at::kFloat) {
    typedef float T;
    typedef float LT;
#include "fused_dense_dropout_layernorm_bwd_tmpl.h"
  } else if (
      inputs[0].dtype() == at::kBFloat16 && inputs[3].dtype() == at::kFloat) {
    typedef bfloat16 T;
    typedef float LT;
#include "fused_dense_dropout_layernorm_bwd_tmpl.h"
  } else if (
      inputs[0].dtype() == at::kBFloat8 && inputs[3].dtype() == at::kFloat) {
    typedef bfloat8 T;
    typedef float LT;
#include "fused_dense_dropout_layernorm_bwd_tmpl.h"
  } else if (
      inputs[0].dtype() == at::kBFloat16 &&
      inputs[3].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
    typedef bfloat16 LT;
#include "fused_dense_dropout_layernorm_bwd_tmpl.h"
  } else if (
      inputs[0].dtype() == at::kBFloat8 && inputs[3].dtype() == at::kBFloat8) {
    typedef bfloat8 T;
    typedef bfloat8 LT;
#include "fused_dense_dropout_layernorm_bwd_tmpl.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

static std::vector<at::Tensor> fused_dense_gelu_fwd(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias,
    bool training) {
  GlobalPass _gp(FWD);
  if (t_in.dtype() == at::kFloat) {
    typedef float T;
#include "fused_dense_gelu_fwd_tmpl.h"
  } else if (t_in.dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "fused_dense_gelu_fwd_tmpl.h"
  } else if (t_in.dtype() == at::kBFloat8) {
    typedef bfloat8 T;
#include "fused_dense_gelu_fwd_tmpl.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

static std::vector<at::Tensor> fused_dense_gelu_bwd(
    at::Tensor t_grad_out,
    at::Tensor t_gelu_in,
    at::Tensor t_in,
    at::Tensor t_wt) {
  GlobalPass _gp(BWD);
  if (t_grad_out.dtype() == at::kFloat) {
    typedef float T;
#include "fused_dense_gelu_bwd_tmpl.h"
  } else if (t_grad_out.dtype() == at::kBFloat16) {
    typedef bfloat16 T;
#include "fused_dense_gelu_bwd_tmpl.h"
  } else if (t_grad_out.dtype() == at::kBFloat8) {
    typedef bfloat8 T;
#include "fused_dense_gelu_bwd_tmpl.h"
  } else {
    TPP_ASSERT(0, "%s:%d Unsupported type\n", __FILE__, __LINE__);
  }
}

static std::vector<at::Tensor> fused_embedding_layernorm_dropout_fwd(
    float p,
    float eps,
    long H,
    long pad_id,
    std::vector<at::Tensor>& inputs,
    bool training) {
  GlobalPass _gp(FWD);
  if (inputs[4].dtype() == at::kFloat && inputs[6].dtype() == at::kFloat) {
    typedef float T;
    typedef float ET;
#include "fused_embedding_layernorm_dropout_fwd_tmpl.h"
  } else if (
      inputs[4].dtype() == at::kBFloat16 && inputs[6].dtype() == at::kFloat) {
    typedef bfloat16 T;
    typedef float ET;
#include "fused_embedding_layernorm_dropout_fwd_tmpl.h"
  } else if (
      inputs[4].dtype() == at::kFloat && inputs[6].dtype() == at::kBFloat16) {
    typedef float T;
    typedef bfloat16 ET;
#include "fused_embedding_layernorm_dropout_fwd_tmpl.h"
  } else if (
      inputs[4].dtype() == at::kBFloat16 &&
      inputs[6].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
    typedef bfloat16 ET;
#include "fused_embedding_layernorm_dropout_fwd_tmpl.h"
  } else if (
      inputs[4].dtype() == at::kBFloat8 && inputs[6].dtype() == at::kFloat) {
    typedef bfloat8 T;
    typedef float ET;
#include "fused_embedding_layernorm_dropout_fwd_tmpl.h"
  } else if (
      inputs[4].dtype() == at::kBFloat8 && inputs[6].dtype() == at::kBFloat8) {
    typedef bfloat8 T;
    typedef bfloat8 ET;
#include "fused_embedding_layernorm_dropout_fwd_tmpl.h"
  } else if (
      inputs[4].dtype() == at::kFloat && inputs[6].dtype() == at::kBFloat8) {
    typedef float T;
    typedef bfloat8 ET;
#include "fused_embedding_layernorm_dropout_fwd_tmpl.h"
  } else {
    TPP_ASSERT(0, "Should not come here\n");
  }
}

static std::vector<at::Tensor> fused_embedding_layernorm_dropout_bwd(
    float p,
    long pad_id,
    std::vector<at::Tensor>& inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat && inputs[6].dtype() == at::kFloat) {
    typedef float T;
    typedef float ET;
#include "fused_embedding_layernorm_dropout_bwd_tmpl.h"
  } else if (
      inputs[0].dtype() == at::kBFloat16 && inputs[6].dtype() == at::kFloat) {
    typedef bfloat16 T;
    typedef float ET;
#include "fused_embedding_layernorm_dropout_bwd_tmpl.h"
  } else if (
      inputs[0].dtype() == at::kFloat && inputs[6].dtype() == at::kBFloat16) {
    typedef float T;
    typedef bfloat16 ET;
#include "fused_embedding_layernorm_dropout_bwd_tmpl.h"
  } else if (
      inputs[0].dtype() == at::kBFloat16 &&
      inputs[6].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
    typedef bfloat16 ET;
#include "fused_embedding_layernorm_dropout_bwd_tmpl.h"
  } else {
    TPP_ASSERT(0, "Should not come here\n");
  }
}

REGISTER_SUBMODULE(_fused_bert_unpad, m) {
  m.def(
      "fused_self_attention_fwd",
      &fused_self_attention_fwd,
      "Tpp BERT forward");
  m.def(
      "fused_self_attention_bwd",
      &fused_self_attention_bwd,
      "Tpp BERT backward");
  m.def(
      "fused_dense_dropout_layernorm_fwd",
      &fused_dense_dropout_layernorm_fwd,
      "Tpp BERT forward");
  m.def(
      "fused_dense_dropout_layernorm_bwd",
      &fused_dense_dropout_layernorm_bwd,
      "Tpp BERT forward");
  m.def("fused_dense_gelu_fwd", &fused_dense_gelu_fwd, "Tpp BERT forward");
  m.def("fused_dense_gelu_bwd", &fused_dense_gelu_bwd, "Tpp BERT forward");
  m.def(
      "fused_embedding_layernorm_dropout_fwd",
      &fused_embedding_layernorm_dropout_fwd,
      "Tpp BERT forward");
  m.def(
      "fused_embedding_layernorm_dropout_bwd",
      &fused_embedding_layernorm_dropout_bwd,
      "Tpp BERT backward");
}

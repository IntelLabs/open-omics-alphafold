/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#ifndef _TENSOR_HELPER_H_
#define _TENSOR_HELPER_H_

#include "utils.h"

template <typename T>
inline at::Tensor wt_tensor_n2v(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  const int BS = get_vnni_block_size<T>();
#if 0
  TPP_ASSERT(Hc % BS == 0, "Uneven number for Hc\n");
  return input.view({Nk, Nc, Hc/BS, BS, Hk}).permute({0, 1, 2, 4, 3}).contiguous();
#else
  auto Hcp2 = (Hc + BS - 1) / BS;
  auto output = input.new_empty({Nk, Nc, Hcp2, Hk, BS});
  auto out = GetVLAPtr<T>(output, {Hcp2 * Hk * BS});
  auto in = GetVLAPtr<T>(input, {Hc * Hk});
  auto n2v_tpp = SCOPEIT(
      XformExtTPP<T>(Hc, Hk, Hcp2 * BS, Hk, XformTPP::XFORM_N2V_TPP), VNNI);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
  for (int n = 0; n < Nk * Nc; n++) {
    n2v_tpp(in[n], out[n]);
  }
  return output;
#endif
}

template <typename T>
inline at::Tensor wt_tensor_trans_n2v(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  const int BS = get_vnni_block_size<T>();
#if 0
  TPP_ASSERT(Hk % BS == 0, "Uneven number for Hk\n");
  return input.view({Nk, Nc, Hc, Hk/BS, BS}).permute({0, 1, 3, 2, 4}).contiguous();
#else
  auto Hkp2 = (Hk + BS - 1) / BS;
  auto output = input.new_empty({Nk, Nc, Hkp2, Hc, BS});
  auto out = GetVLAPtr<T>(output, {Hkp2 * Hc * BS});
  auto in = GetVLAPtr<T>(input, {Hc * Hk});
  auto trans_n2v_tpp = SCOPEIT(
      XformExtTPP<T>(
          Hc, Hk, Hc, Hkp2 * BS, XformTPP::XFORM_XPOSE_N2V_TPP, true),
      XPOSE);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
  for (int n = 0; n < Nk * Nc; n++) {
    trans_n2v_tpp(in[n], out[n]);
  }
  return output;
#endif
}

template <typename T>
inline at::Tensor wt_tensor_trans_n2v_compact(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  const int BS = get_vnni_block_size<T>();
#if 0
  TPP_ASSERT(Hk % BS == 0, "Uneven number for Hk\n");
  return input.view({Nk, Nc, Hc, Hk/BS, BS}).permute({1, 0, 3, 2, 4}).contiguous();
#else
  auto Hkp2 = (Hk + BS - 1) / BS;
  auto output = input.new_empty({Nc, Nk, Hkp2, Hc, BS});
  auto out = GetVLAPtr<T>(output, {Nk, Hkp2 * Hc * BS});
  auto in = GetVLAPtr<T>(input, {Nc, Hc * Hk});
  auto trans_n2v_tpp = SCOPEIT(
      XformExtTPP<T>(
          Hc, Hk, Hc, Hkp2 * BS, XformTPP::XFORM_XPOSE_N2V_TPP, true),
      XPOSE);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
  for (int nk = 0; nk < Nk; nk++) {
    for (int nc = 0; nc < Nc; nc++) {
      trans_n2v_tpp(in[nk][nc], out[nc][nk]);
    }
  }
  return output;
#endif
}

template <typename T>
inline at::Tensor wt_tensor_trans_v2v(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  const int BS = get_vnni_block_size<T>();
#if 0
  TPP_ASSERT(Hc % BS == 0, "Uneven number for Hc\n");
  TPP_ASSERT(Hk % BS == 0, "Uneven number for Hk\n");
  return input.view({Nk, Nc, Hc/BS, Hk/BS, BS, BS}).permute({0, 1, 3, 2, 5, 4}).contiguous().view({Nk, Nc, Hk/BS, Hc, BS});
#else
  TPP_ASSERT(Hc % BS == 0, "Uneven number for Hc\n");
  auto Hkp2 = (Hk + BS - 1) / BS;
  auto output = input.new_empty({Nk, Nc, Hkp2, Hc, BS});
  auto out = GetVLAPtr<T>(output, {Hkp2 * Hc * BS});
  auto in = GetVLAPtr<T>(input, {Hc * Hk});
  auto trans_v2v_tpp = SCOPEIT(
      XformExtTPP<T>(
          Hc, Hk, Hkp2 * BS, Hc, XformTPP::XFORM_XPOSE_V2V_TPP, true),
      XPOSE);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
  for (int n = 0; n < Nk * Nc; n++) {
    trans_v2v_tpp(in[n], out[n]);
  }
  return output;
#endif
}

template <typename T>
inline at::Tensor wt_tensor_trans_v2v_compact(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  const int BS = get_vnni_block_size<T>();
#if 0
  TPP_ASSERT(Hc % BS == 0, "Uneven number for Hc\n");
  TPP_ASSERT(Hk % BS == 0, "Uneven number for Hk\n");
  return input.view({Nk, Nc, Hc/BS, Hk/BS, BS, BS}).permute({1, 0, 3, 2, 5, 4}).contiguous().view({Nc, Nk, Hk/BS, Hc, BS});
#else
  TPP_ASSERT(Hc % BS == 0, "Uneven number for Hc\n");
  auto Hkp2 = (Hk + BS - 1) / BS;
  auto output = input.new_empty({Nc, Nk, Hkp2, Hc, BS});
  auto out = GetVLAPtr<T>(output, {Nk, Hkp2 * Hc * BS});
  auto in = GetVLAPtr<T>(input, {Nc, Hc * Hk});
  auto trans_v2v_tpp = SCOPEIT(
      XformExtTPP<T>(
          Hc, Hk, Hkp2 * BS, Hc, XformTPP::XFORM_XPOSE_V2V_TPP, true),
      XPOSE);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
  for (int nk = 0; nk < Nk; nk++) {
    for (int nc = 0; nc < Nc; nc++) {
      trans_v2v_tpp(in[nk][nc], out[nc][nk]);
    }
  }
  return output;
#endif
}

USING_SCOPE(w_vnni);

inline at::Tensor wt_tensor_for_fwd(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  RECORD_SCOPE(w_vnni, {input});
  if (input.dtype() != at::kFloat) {
    if (input.dim() == 5) {
      return input;
    } else {
      if (input.dtype() == at::kBFloat16) {
        return wt_tensor_n2v<bfloat16>(Nk, Hk, Nc, Hc, input);
      } else if (input.dtype() == at::kBFloat8) {
        return wt_tensor_n2v<bfloat8>(Nk, Hk, Nc, Hc, input);
      } else {
        TPP_ASSERT(false, "Unsupported datatype!");
      }
    }
  } else {
    return input;
  }
}

USING_SCOPE(w_xpose);

inline at::Tensor wt_tensor_for_bwd(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  RECORD_SCOPE(w_xpose, {input});
  if (input.dtype() == at::kBFloat16) {
    if (input.dim() == 5) {
      return wt_tensor_trans_v2v<bfloat16>(Nk, Hk, Nc, Hc, input);
    } else {
      return wt_tensor_trans_n2v<bfloat16>(Nk, Hk, Nc, Hc, input);
    }
  } else if (input.dtype() == at::kBFloat8) {
    if (input.dim() == 5) {
      return wt_tensor_trans_v2v<bfloat8>(Nk, Hk, Nc, Hc, input);
    } else {
      return wt_tensor_trans_n2v<bfloat8>(Nk, Hk, Nc, Hc, input);
    }
  } else if (input.dtype() == at::kFloat) {
#if 0
    return input.permute({0, 1, 3, 2}).contiguous();
#else
    auto output = input.new_empty({Nk, Nc, Hk, Hc});
    auto out = GetVLAPtr<float>(output, {Hk * Hc});
    auto in = GetVLAPtr<float>(input, {Hc * Hk});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<float>(Hc, Hk, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < Nk * Nc; n++) {
      trans_tpp(in[n], out[n]);
    }
    return output;
#endif
  } else {
    TPP_ASSERT(false, "Unsupported datatype!");
  }
}

inline at::Tensor wt_tensor_for_bwd_compact(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  RECORD_SCOPE(w_xpose, {input});
  if (input.dtype() == at::kBFloat16) {
    if (input.dim() == 5) {
      return wt_tensor_trans_v2v_compact<bfloat16>(Nk, Hk, Nc, Hc, input);
    } else {
      return wt_tensor_trans_n2v_compact<bfloat16>(Nk, Hk, Nc, Hc, input);
    }
  } else if (input.dtype() == at::kBFloat8) {
    if (input.dim() == 5) {
      return wt_tensor_trans_v2v_compact<bfloat8>(Nk, Hk, Nc, Hc, input);
    } else {
      return wt_tensor_trans_n2v_compact<bfloat8>(Nk, Hk, Nc, Hc, input);
    }
  } else if (input.dtype() == at::kFloat) {
#if 0
    return input.permute({1, 0, 3, 2}).contiguous();
#else
    auto output = input.new_empty({Nk, Nc, Hk, Hc});
    auto out = GetVLAPtr<float>(output, {Nk, Hk * Hc});
    auto in = GetVLAPtr<float>(input, {Nc, Hc * Hk});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<float>(Hc, Hk, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int nk = 0; nk < Nk; nk++) {
      for (int nc = 0; nc < Nc; nc++) {
        trans_tpp(in[nk][nc], out[nc][nk]);
      }
    }
    return output;
#endif
  } else {
    TPP_ASSERT(false, "Unsupported datatype!");
  }
}

USING_SCOPE(a_xpose);

inline at::Tensor act_tensor_trans(
    long B,
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_xpose, {input});
#if 0
  return input.permute({0, 1, 2, 4, 3}).contiguous();
#else
  auto output = input.new_empty({B, S1, N, H, S2});
  if (input.dtype() == at::kBFloat16) {
    auto out = GetVLAPtr<bfloat16>(output, {H * S2});
    auto in = GetVLAPtr<bfloat16>(input, {H * S2});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < B * S1 * N; n++) {
        trans_tpp(in[n], out[n]);
      }
    }
  } else if (input.dtype() == at::kBFloat8) {
    auto out = GetVLAPtr<bfloat8>(output, {H * S2});
    auto in = GetVLAPtr<bfloat8>(input, {H * S2});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<bfloat8>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < B * S1 * N; n++) {
        trans_tpp(in[n], out[n]);
      }
    }
  } else {
    TPP_ASSERT(false, "Unsupported datatype!");
  }
  return output;
#endif
}

inline at::Tensor act_tensor_trans(
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_xpose, {input});
#if 0
  return input.permute({0, 1, 3, 2}).contiguous();
#else
  auto output = input.new_empty({S1, N, H, S2});
  if (input.dtype() == at::kBFloat16) {
    auto out = GetVLAPtr<bfloat16>(output, {H * S2});
    auto in = GetVLAPtr<bfloat16>(input, {H * S2});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < S1 * N; n++) {
        trans_tpp(in[n], out[n]);
      }
    }
  } else if (input.dtype() == at::kBFloat8) {
    auto out = GetVLAPtr<bfloat8>(output, {H * S2});
    auto in = GetVLAPtr<bfloat8>(input, {H * S2});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<bfloat8>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < S1 * N; n++) {
        trans_tpp(in[n], out[n]);
      }
    }
  } else {
    TPP_ASSERT(false, "Unsupported datatype!");
  }
  return output;
#endif
}

inline at::Tensor act_tensor_trans_compact(
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_xpose, {input});
#if 0
  return input.permute({1, 0, 3, 2}).contiguous();
#else
  auto output = input.new_empty({N, S1, H, S2});
  if (input.dtype() == at::kBFloat16) {
    auto out = GetVLAPtr<bfloat16>(output, {S1, H * S2});
    auto in = GetVLAPtr<bfloat16>(input, {N, H * S2});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int n = 0; n < N; n++) {
          trans_tpp(in[s1][n], out[n][s1]);
        }
      }
    }
  } else if (input.dtype() == at::kBFloat8) {
    auto out = GetVLAPtr<bfloat8>(output, {S1, H * S2});
    auto in = GetVLAPtr<bfloat8>(input, {N, H * S2});
    auto trans_tpp =
        SCOPEIT(XformExtTPP<bfloat8>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int n = 0; n < N; n++) {
          trans_tpp(in[s1][n], out[n][s1]);
        }
      }
    }
  } else {
    TPP_ASSERT(false, "Unsupported datatype!");
  }
  return output;
#endif
}

USING_SCOPE(a_vnni);

inline at::Tensor act_tensor_n2v(
    long B,
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_vnni, {input});
  const int BS = get_vnni_block_size(input.dtype());
  TPP_ASSERT(S2 % BS == 0, "Uneven number for S2\n");
#if 1
  return input.view({B, S1, N, S2 / BS, BS, H})
      .permute({0, 1, 2, 3, 5, 4})
      .contiguous();
#else
  auto output = input.new_empty({B, S1, N, S2 / BS, H, BS});
  auto out = GetVLAPtr<bfloat16>(output, {H * S2});
  auto in = GetVLAPtr<bfloat16>(input, {H * S2});
  auto n2v_tpp =
      SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_N2V_TPP), VNNI);
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < B * S1 * N; n++) {
      n2v_tpp(in[n], out[n]);
    }
  }
  return output;
#endif
}

inline at::Tensor act_tensor_n2v(
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_vnni, {input});
  const int BS = get_vnni_block_size(input.dtype());
  TPP_ASSERT(S2 % BS == 0, "Uneven number for S2\n");
#if 1
  return input.view({S1, N, S2 / BS, BS, H})
      .permute({0, 1, 2, 4, 3})
      .contiguous();
#else
  auto output = input.new_empty({S1, N, S2 / BS, H, BS});
  auto out = GetVLAPtr<bfloat16>(output, {H * S2});
  auto in = GetVLAPtr<bfloat16>(input, {H * S2});
  auto n2v_tpp =
      SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_N2V_TPP), VNNI);
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < S1 * N; n++) {
      n2v_tpp(in[n], out[n]);
    }
  }
  return output;
#endif
}

inline at::Tensor act_tensor_n2v_compact(
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_vnni, {input});
  const int BS = get_vnni_block_size(input.dtype());
  TPP_ASSERT(S2 % BS == 0, "Uneven number for S2\n");
#if 0
  return input.view({S1, N, S2/BS, BS, H}).permute({1,0,2,4,3}).contiguous();
#else
  auto output = input.new_empty({N, S1, S2 / BS, H, BS});
  if (input.dtype() == at::kBFloat16) {
    auto out = GetVLAPtr<bfloat16>(output, {S1, H * S2});
    auto in = GetVLAPtr<bfloat16>(input, {N, H * S2});
    auto n2v_tpp =
        SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_N2V_TPP), VNNI);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int n = 0; n < N; n++) {
          n2v_tpp(in[s1][n], out[n][s1]);
        }
      }
    }
  } else if (input.dtype() == at::kBFloat8) {
    auto out = GetVLAPtr<bfloat8>(output, {S1, H * S2});
    auto in = GetVLAPtr<bfloat8>(input, {N, H * S2});
    auto n2v_tpp =
        SCOPEIT(XformExtTPP<bfloat8>(S2, H, XformTPP::XFORM_N2V_TPP), VNNI);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int s1 = 0; s1 < S1; s1++) {
        for (int n = 0; n < N; n++) {
          n2v_tpp(in[s1][n], out[n][s1]);
        }
      }
    }
  } else {
    TPP_ASSERT(false, "Unsupported datatype!");
  }

  return output;
#endif
}

inline at::Tensor get_padded_activation_for_vnni(at::Tensor& input) {
  RECORD_SCOPE(a_vnni, {input});
  auto dtype = input.dtype();
  if (dtype == at::kFloat)
    return input;
  const int align = get_vnni_block_size(dtype);
  auto sizes = input.sizes();
  int ndims = input.dim();
  TPP_ASSERT(ndims >= 2, "Invalid shape\n");
  auto C = sizes[ndims - 1];
  int pad = C % align;
  if (pad == 0)
    return input;
  std::vector<int64_t> new_sizes(sizes.begin(), sizes.end());
  new_sizes[ndims - 1] = align - pad;
  auto output = at::cat({input, input.new_zeros(new_sizes)}, ndims - 1);
  return output;
}

USING_SCOPE(zero);

inline void tensor_set_zero(long N, long sz, at::Tensor& input) {
#if 0
  input.zero_();
#else
  RECORD_SCOPE(zero, {input});
  if (input.dtype() == at::kFloat) {
    auto in = GetVLAPtr<float>(input, {sz});
    auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(sz), EW_ZERO);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      set_zero_tpp(in[n]);
    }
  } else if (input.dtype() == at::kBFloat16) {
    auto in = GetVLAPtr<bfloat16>(input, {sz});
    auto set_zero_tpp = SCOPEIT(SetZeroTPP<bfloat16>(sz), EW_ZERO);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      set_zero_tpp(in[n]);
    }
  } else {
    auto in = GetVLAPtr<bfloat8>(input, {sz});
    auto set_zero_tpp = SCOPEIT(SetZeroTPP<bfloat8>(sz), EW_ZERO);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      set_zero_tpp(in[n]);
    }
  }
#endif
}

template <typename T>
class LToPBlockAccessMapper {
 public:
  LToPBlockAccessMapper(long M, long N) : M(M), N(N) {}
  long operator()(long i, long j) {
    if (std::is_same<T, float>()) {
      return i * N + j;
    } else {
      return j * M + i;
    }
  }

 private:
  long M, N;
};

template <typename Tin, typename Tout>
inline void omp_reduce_buf(
    int num_threads,
    int N,
    Tout** ptrs,
    Tin* buf,
    bool accumulate = false) {
  ScopedTimer _t(EW_RED);
#pragma omp for
  for (int i = 0; i < N; i++) {
    float sum = 0.0;
    for (int j = 0; j < num_threads; j++) {
      sum += ptrs[j][i];
    }
    if (accumulate) {
      buf[i] += sum;
    } else {
      buf[i] = sum;
    }
  }
}

#endif // _TENSOR_HELPER_H_

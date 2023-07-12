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
#include "bfloat8.h"
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
#include "tensor_helper.h"

template <typename Tin, typename Tout>
void cvt_impl(long N, Tin* inp, Tout* outp) {
  constexpr int BS = 256;
  auto cvt_tpp = SCOPEIT((ConvertTPP<Tin, Tout>(BS)), EW_COPY);
  long i;
#pragma omp parallel for lastprivate(i)
  for (i = 0; i < ALIGNDOWN(N, BS); i += BS) {
    cvt_tpp(&inp[i], &outp[i]);
  }
  if (i < N) {
    auto cvt_tpp = SCOPEIT((ConvertTPP<Tin, Tout>(N - i)), EW_COPY);
    cvt_tpp(&inp[i], &outp[i]);
  }
}

static at::Tensor cvt_to_bf8(at::Tensor self) {
  auto out = at::empty_like(self, at::kBFloat8);
  long N = self.numel();
  if (self.dtype() == at::kFloat) {
    cvt_impl(N, pt_get_data_ptr<float>(self), pt_get_data_ptr<bfloat8>(out));
  } else if (self.dtype() == at::kBFloat16) {
    cvt_impl(N, pt_get_data_ptr<bfloat16>(self), pt_get_data_ptr<bfloat8>(out));
  } else if (self.dtype() == at::kHalf) {
    cvt_impl(N, pt_get_data_ptr<half>(self), pt_get_data_ptr<bfloat8>(out));
  } else {
    TPP_ASSERT(false, "Unsupported datatype for cvt_to_bf8\n");
  }
  return out;
}

static at::Tensor cvt_from_bf8(at::Tensor self, py::object dtype_) {
  TPP_ASSERT(self.dtype() == at::kBFloat8, "Input must be of BFloat8 datatype");
  at::ScalarType dtype = torch::python::detail::py_object_to_dtype(dtype_);
  auto out = at::empty_like(self, dtype);
  long N = self.numel();
  if (dtype == at::kFloat) {
    cvt_impl(N, pt_get_data_ptr<bfloat8>(self), pt_get_data_ptr<float>(out));
  } else if (dtype == at::kBFloat16) {
    cvt_impl(N, pt_get_data_ptr<bfloat8>(self), pt_get_data_ptr<bfloat16>(out));
  } else if (dtype == at::kHalf) {
    cvt_impl(N, pt_get_data_ptr<bfloat8>(self), pt_get_data_ptr<half>(out));
  } else {
    TPP_ASSERT(false, "Unsupported datatype for cvt_from_bf8\n");
  }
  return out;
}

REGISTER_SUBMODULE(_bf8, m) {
  m.def("cvt_to_bf8", &cvt_to_bf8, "Convert to Bfloat8");
  m.def("cvt_from_bf8", &cvt_from_bf8, "Convert from Bfloat8 to dtype");
}

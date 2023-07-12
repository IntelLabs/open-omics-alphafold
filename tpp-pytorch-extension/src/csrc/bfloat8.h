/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#ifndef _TPP_BFLOAT8_H_
#define _TPP_BFLOAT8_H_

#ifndef PYTORCH_SUPPORTS_BFLOAT8
#include <libxsmm.h>

namespace at {

constexpr auto kBFloat8 = kByte;
class BFloat8 {
 public:
  BFloat8() {}
  BFloat8(float f) {
    libxsmm_rne_convert_fp32_bf8(&f, &val, 1);
  }
  operator float() {
    float f;
    libxsmm_convert_bf8_f32(&val, &f, 1);
    return f;
  }

  template <typename T>
  BFloat8& operator+=(const T& rhs) {
    *this = float(*this) + float(rhs);
    return *this;
  }

 protected:
  libxsmm_bfloat8 val;
};

}; // namespace at

#endif // PYTORCH_SUPPORTS_BFLOAT8

#endif // _TPP_BFLOAT8_H_

###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)                                       #
###############################################################################

import torch
from .._C import _bf8 as bf8_cpp

pytorch_supports_bfloat8 = hasattr(torch, "bfloat8") and torch.bfloat8 != torch.uint8
if not pytorch_supports_bfloat8:
    torch.bfloat8 = torch.uint8


def cvt_to(self, dtype):
    # print(f"CVT_TO called: self.dtype: {self.dtype}, dtype: {dtype}")
    if self.dtype == dtype:
        return self
    if pytorch_supports_bfloat8 or (
        dtype != torch.bfloat8 and self.dtype != torch.bfloat8
    ):
        return self.to(dtype)
    else:
        # print(f"CVT_TO called: self.dtype: {self.dtype}, dtype: {dtype} shape: {list(self.shape)}")
        if dtype == torch.bfloat8:
            return bf8_cpp.cvt_to_bf8(self)
        else:
            return bf8_cpp.cvt_from_bf8(self, dtype)


torch.Tensor.cvt_to = cvt_to

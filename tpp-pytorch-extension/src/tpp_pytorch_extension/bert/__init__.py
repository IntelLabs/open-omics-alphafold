###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)                                       #
###############################################################################

from . import fused_bert_unpad
from . import fused_bert
from tpp_pytorch_extension import manual_seed
from tpp_pytorch_extension import reset_debug_timers
from tpp_pytorch_extension import print_debug_timers
from tpp_pytorch_extension import print_debug_thread_imbalance
from tpp_pytorch_extension.optim import AdamW
from tpp_pytorch_extension.optim import Lamb
from tpp_pytorch_extension.optim import DistLamb
from tpp_pytorch_extension.optim import clip_grad_norm_
from tpp_pytorch_extension.utils.blocked_layout import block_model_params as block
from contextlib import contextmanager


@contextmanager
def tpp_impl(enable=True, use_low_prec=False, use_unpad=True, use_bf8=False):
    if use_unpad == True:
        with fused_bert_unpad.tpp_impl(enable, use_low_prec, use_bf8):
            yield
    else:
        if use_low_prec and use_bf8:
            raise NotImplementedError("BF8 is only supported with unpad")
        with fused_bert.tpp_impl(enable, use_low_prec):
            yield


def set_rnd_seed(seed):
    manual_seed(seed)

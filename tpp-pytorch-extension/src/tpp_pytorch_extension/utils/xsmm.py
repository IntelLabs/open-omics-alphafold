###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)                                       #
###############################################################################

from .._C import _xsmm as xsmm_cpp


def manual_seed(seed):
    xsmm_cpp.manual_seed(seed)


def set_rng_state(new_state):
    raise NotImplementedError


def get_rng_state():
    raise NotImplementedError


def get_vnni_blocking(dtype):
    return xsmm_cpp.get_vnni_blocking(dtype)


# initialize libxsmm library and random number generator
xsmm_cpp.init_libxsmm()
manual_seed(12345)

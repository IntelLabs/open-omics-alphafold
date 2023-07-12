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

from . import _C

try:
    from .utils import extend_profiler
except:
    extend_profiler = None

from .utils.xsmm import manual_seed
from .utils.xsmm import get_vnni_blocking
from .utils import blocked_layout
from .utils import bfloat8
from . import optim


def reset_debug_timers():
    _C.reset_debug_timers()


def print_debug_timers(tid=0):
    _C.print_debug_timers(tid)


def print_debug_thread_imbalance():
    _C.print_debug_thread_imbalance()


reset_debug_timers()

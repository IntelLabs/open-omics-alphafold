###############################################################################
# Copyright (c) 2023 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Narendra Chaudhary (Intel Corp.)                                       #
###############################################################################


import math
import torch
from torch import nn
from torch.autograd import Function

from tpp_pytorch_extension._C import (
    _alpha_attention as Alpha_TriangleMultiplication_cpp,
)


class TriangleMultiplicationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        act,
        mask,
        c_equation,
        layer_norm_input_weight,
        layer_norm_input_bias,
        left_projection_weight,
        left_projection_bias,
        right_projection_weight,
        right_projection_bias,
        left_gate_weight,
        left_gate_bias,
        right_gate_weight,
        right_gate_bias,
        center_layer_norm_weight,
        center_layer_norm_bias,
        output_projection_weight,
        output_projection_bias,
        gating_linear_weight,
        gating_linear_bias,
    ):
        equation_flag = int(0)
        if c_equation == "ikc,jkc->ijc":  # "Outgoing" edges equation
            equation_flag = 0
        else:  # "Incoming" edges equation
            equation_flag = 1
        act = Alpha_TriangleMultiplication_cpp.trianglemulti_forward(
            act.contiguous(),
            mask.contiguous(),
            equation_flag,
            layer_norm_input_weight,
            layer_norm_input_bias,
            left_projection_weight,
            left_projection_bias,
            right_projection_weight,
            right_projection_bias,
            left_gate_weight,
            left_gate_bias,
            right_gate_weight,
            right_gate_bias,
            center_layer_norm_weight,
            center_layer_norm_bias,
            output_projection_weight,
            output_projection_bias,
            gating_linear_weight,
            gating_linear_bias,
        )
        return act


def TriangleMultiplicationOpti_forward(self, act, mask):
    mask = mask[..., None]
    if (
        act.dtype == torch.bfloat16
        or mask.dtype == torch.bfloat16
        or self.layer_norm_input.weight.dtype == torch.bfloat16
        or self.layer_norm_input.bias.dtype == torch.bfloat16
        or self.left_projection.weight.dtype == torch.bfloat16
        or self.left_projection.bias.dtype == torch.bfloat16
        or self.right_projection.weight.dtype == torch.bfloat16
        or self.right_projection.bias.dtype == torch.bfloat16
        or self.left_gate.weight.dtype == torch.bfloat16
        or self.left_gate.bias.dtype == torch.bfloat16
        or self.right_gate.weight.dtype == torch.bfloat16
        or self.right_gate.bias.dtype == torch.bfloat16
        or self.center_layer_norm.weight.dtype == torch.bfloat16
        or self.center_layer_norm.bias.dtype == torch.bfloat16
        or self.output_projection.weight.dtype == torch.bfloat16
        or self.output_projection.bias.dtype == torch.bfloat16
        or self.gating_linear.weight.dtype == torch.bfloat16
        or self.gating_linear.bias.dtype == torch.bfloat16
    ):
        act = TriangleMultiplicationFunction.apply(
            act.to(torch.bfloat16),
            mask.to(torch.float32),
            self.c_equation,
            self.layer_norm_input.weight.to(torch.bfloat16),
            self.layer_norm_input.bias.to(torch.bfloat16),
            self.left_projection.weight.to(torch.bfloat16),
            self.left_projection.bias.to(torch.float32),
            self.right_projection.weight.to(torch.bfloat16),
            self.right_projection.bias.to(torch.float32),
            self.left_gate.weight.to(torch.bfloat16),
            self.left_gate.bias.to(torch.float32),
            self.right_gate.weight.to(torch.bfloat16),
            self.right_gate.bias.to(torch.float32),
            self.center_layer_norm.weight.to(torch.bfloat16),
            self.center_layer_norm.bias.to(torch.bfloat16),
            self.output_projection.weight.to(torch.bfloat16),
            self.output_projection.bias.to(torch.float32),
            self.gating_linear.weight.to(torch.bfloat16),
            self.gating_linear.bias.to(torch.float32),
        )
    else:
        act = TriangleMultiplicationFunction.apply(
            act,
            mask,
            self.c_equation,
            self.layer_norm_input.weight,
            self.layer_norm_input.bias,
            self.left_projection.weight,
            self.left_projection.bias,
            self.right_projection.weight,
            self.right_projection.bias,
            self.left_gate.weight,
            self.left_gate.bias,
            self.right_gate.weight,
            self.right_gate.bias,
            self.center_layer_norm.weight,
            self.center_layer_norm.bias,
            self.output_projection.weight,
            self.output_projection.bias,
            self.gating_linear.weight,
            self.gating_linear.bias,
        )
    return act


class TriangleMultiplicationOpti(nn.Module):

    #   def __init__(self,config, global_config, act_dim):
    def __init__(self, equation, num_intermediate_channel, act_dim):
        """Builds TriangleMultiplication module.

        Arguments:
          act: Pair activations, shape [N_res, N_res, c_z]
          mask: Pair mask, shape [N_res, N_res].
          is_training: Whether the module is in training mode.

        Returns:
          Outputs, same shape/type as act.
        """
        super().__init__()
        # self.config = config
        # self.global_config = global_config
        # self.c_equation = self.config['equation']
        self.c_equation = equation
        # self.num_intermediate_channel = num_intermediate_channel
        self.layer_norm_input = nn.LayerNorm(
            normalized_shape=act_dim, elementwise_affine=True
        )
        self.left_projection = nn.Linear(act_dim, num_intermediate_channel)
        self.right_projection = nn.Linear(act_dim, num_intermediate_channel)
        self.left_gate = nn.Linear(act_dim, num_intermediate_channel)
        self.right_gate = nn.Linear(act_dim, num_intermediate_channel)
        self.center_layer_norm = nn.LayerNorm(
            normalized_shape=act_dim, elementwise_affine=True
        )
        self.output_projection = nn.Linear(act_dim, act_dim)
        self.gating_linear = nn.Linear(act_dim, act_dim)

    def forward(self, act, mask):
        mask = mask[..., None]
        # act = self.layer_norm_input(act)
        # input_act = act # For gate
        # left_proj_act = mask * self.left_projection(act)
        # right_proj_act = mask * self.right_projection(act)
        # left_proj_act *= torch.sigmoid(self.left_gate(act))
        # right_proj_act *= torch.sigmoid(self.right_gate(act))
        # # "Outgoing" edges equation: 'ikc,jkc->ijc'
        # # "Incoming" edges equation: 'kjc,kic->ijc'
        # # Note on the Suppl. Alg. 11 & 12 notation:
        # # For the "outgoing" edges, a = left_proj_act and b = right_proj_act
        # # For the "incoming" edges, it's swapped:
        # #   b = left_proj_act and a = right_proj_act
        # act = torch.einsum(self.c_equation, left_proj_act, right_proj_act)
        # act = self.center_layer_norm(act)
        # act = self.output_projection(act)
        # act *= torch.sigmoid(self.gating_linear(input_act))
        if (
            act.dtype == torch.bfloat16
            or mask.dtype == torch.bfloat16
            or self.layer_norm_input.weight.dtype == torch.bfloat16
            or self.layer_norm_input.bias.dtype == torch.bfloat16
            or self.left_projection.weight.dtype == torch.bfloat16
            or self.left_projection.bias.dtype == torch.bfloat16
            or self.right_projection.weight.dtype == torch.bfloat16
            or self.right_projection.bias.dtype == torch.bfloat16
            or self.left_gate.weight.dtype == torch.bfloat16
            or self.left_gate.bias.dtype == torch.bfloat16
            or self.right_gate.weight.dtype == torch.bfloat16
            or self.right_gate.bias.dtype == torch.bfloat16
            or self.center_layer_norm.weight.dtype == torch.bfloat16
            or self.center_layer_norm.bias.dtype == torch.bfloat16
            or self.output_projection.weight.dtype == torch.bfloat16
            or self.output_projection.bias.dtype == torch.bfloat16
            or self.gating_linear.weight.dtype == torch.bfloat16
            or self.gating_linear.bias.dtype == torch.bfloat16
        ):
            act = TriangleMultiplicationFunction.apply(
                act.to(torch.bfloat16),
                mask.to(torch.float32),
                self.c_equation,
                self.layer_norm_input.weight.to(torch.bfloat16),
                self.layer_norm_input.bias.to(torch.bfloat16),
                self.left_projection.weight.to(torch.bfloat16),
                self.left_projection.bias.to(torch.float32),
                self.right_projection.weight.to(torch.bfloat16),
                self.right_projection.bias.to(torch.float32),
                self.left_gate.weight.to(torch.bfloat16),
                self.left_gate.bias.to(torch.float32),
                self.right_gate.weight.to(torch.bfloat16),
                self.right_gate.bias.to(torch.float32),
                self.center_layer_norm.weight.to(torch.bfloat16),
                self.center_layer_norm.bias.to(torch.bfloat16),
                self.output_projection.weight.to(torch.bfloat16),
                self.output_projection.bias.to(torch.float32),
                self.gating_linear.weight.to(torch.bfloat16),
                self.gating_linear.bias.to(torch.float32),
            )
        else:
            act = TriangleMultiplicationFunction.apply(
                act,
                mask,
                self.c_equation,
                self.layer_norm_input.weight,
                self.layer_norm_input.bias,
                self.left_projection.weight,
                self.left_projection.bias,
                self.right_projection.weight,
                self.right_projection.bias,
                self.left_gate.weight,
                self.left_gate.bias,
                self.right_gate.weight,
                self.right_gate.bias,
                self.center_layer_norm.weight,
                self.center_layer_norm.bias,
                self.output_projection.weight,
                self.output_projection.bias,
                self.gating_linear.weight,
                self.gating_linear.bias,
            )
        return act

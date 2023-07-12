###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Narendra Chaudhary (Intel Corp.)                                       #
###############################################################################

import time
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math

import tpp_pytorch_extension
from tpp_pytorch_extension.alphafold.Alpha_Attention import GatingAttentionOpti

torch.set_default_tensor_type(torch.FloatTensor)


class GatingAttention(nn.Module):
    """Multihead attention w/ Gating"""

    def __init__(self, num_head, a_dim, m_dim, output_dim):
        super().__init__()
        # self.config = config
        # self.global_config = global_config
        self.output_dim = output_dim
        # k,v dim
        self.key_dim = int(a_dim)
        self.value_dim = int(m_dim)
        self.num_head = num_head
        assert self.key_dim % self.num_head == 0
        assert self.value_dim % self.num_head == 0
        self.key_dim = self.key_dim // self.num_head
        self.value_dim = self.value_dim // self.num_head
        # q,k,v weights
        self.query_w = nn.Parameter(
            torch.Tensor(a_dim, self.num_head, self.key_dim), requires_grad=False
        )
        self.key_w = nn.Parameter(
            torch.Tensor(m_dim, self.num_head, self.key_dim), requires_grad=False
        )
        self.value_w = nn.Parameter(
            torch.Tensor(m_dim, self.num_head, self.value_dim), requires_grad=False
        )
        self.gating_w = nn.Parameter(
            torch.Tensor(a_dim, self.num_head, self.value_dim), requires_grad=False
        )
        self.gating_b = nn.Parameter(
            torch.Tensor(self.num_head, self.value_dim), requires_grad=False
        )
        self.output_w = nn.Parameter(
            torch.Tensor(self.num_head, self.value_dim, self.output_dim),
            requires_grad=False,
        )
        self.output_b = nn.Parameter(torch.Tensor(self.output_dim), requires_grad=False)
        # softmax & act fn
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, q_data, m_data, bias, nonbatched_bias=torch.Tensor()):
        """Builds Attention module.
        Arguments:
          q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
          m_data: A tensor of memories from which the keys and values are
            projected, shape [batch_size, N_keys, m_channels].
          bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
          nonbatched_bias: Shared bias, shape [N_queries, N_keys].
        Returns:
          A float32 tensor of shape [batch_size, N_queries, output_dim].
        """

        # get query, key, value
        q = torch.einsum("bqa,ahc->bqhc", q_data, self.query_w) * self.key_dim ** (-0.5)
        k = torch.einsum("bka,ahc->bkhc", m_data, self.key_w)
        v = torch.einsum("bka,ahc->bkhc", m_data, self.value_w)

        logits = torch.einsum("bqhc,bkhc->bhqk", q, k) + bias

        if nonbatched_bias.shape[0] > 0:
            logits += torch.unsqueeze(nonbatched_bias, dim=0)
        weights = self.softmax(logits)

        weighted_avg = torch.einsum("bhqk,bkhc->bqhc", weights, v)

        gate_values = (
            torch.einsum("bqc,chv->bqhv", q_data, self.gating_w) + self.gating_b
        )
        gate_values = self.sigmoid(gate_values)
        weighted_avg *= gate_values

        output = (
            torch.einsum("bqhc,hco->bqo", weighted_avg, self.output_w) + self.output_b
        )
        return output


set = 1

if set == 1:
    B, S, HS = 512, 764, 256
    N, H = 8, 32

if set == 2:
    B, S, HS = 764, 764, 64
    N, H = 4, 16

if set == 3:
    B, S, HS = 320, 764, 64
    N, H = 8, 8

if set == 4:
    B, S, HS = 764, 764, 128
    N, H = 4, 32

if set == 5:
    B, S, HS = 764, 512, 256
    N, H = 8, 32


class Net1(nn.Module):  # First network containing original attention layer
    def __init__(self):
        super(Net1, self).__init__()
        self.attention = GatingAttention(
            num_head=N, a_dim=HS, m_dim=HS, output_dim=HS
        )  # Attention layer

    def forward(self, q_data, m_data, bias, nonbatched_bias):
        x = self.attention(q_data, m_data, bias, nonbatched_bias)
        return x


class Net2(nn.Module):  # Second network containing optimized attention layer
    def __init__(self):
        super(Net2, self).__init__()
        self.attention = GatingAttentionOpti(
            num_head=N, a_dim=HS, m_dim=HS, output_dim=HS
        )  # Attention layer

    def forward(self, q_data, m_data, bias, nonbatched_bias):
        x = self.attention(q_data, m_data, bias, nonbatched_bias)
        return x


net1 = Net1()
net2 = Net2()

torch.manual_seed(11)  # Set random seed for reproducibility

q_data = torch.randn(B, S, HS, requires_grad=False)
m_data = torch.randn(B, S, HS, requires_grad=False)
bias = torch.randn(B, 1, 1, S, requires_grad=False)
nonbatched_bias = torch.randn(N, S, S, requires_grad=False)
# nonbatched_bias = torch.Tensor()

query_w = torch.randn(HS, N, H)
key_w = torch.randn(HS, N, H)
value_w = torch.randn(HS, N, H)
gating_w = torch.randn(HS, N, H)
gating_b = torch.randn(N, H)
output_w = torch.randn(N, H, HS)
output_b = torch.randn(HS)

net1.attention.query_w.data = query_w
net1.attention.key_w.data = key_w
net1.attention.value_w.data = value_w
net1.attention.gating_w.data = gating_w
net1.attention.gating_b.data = gating_b
net1.attention.output_w.data = output_w
net1.attention.output_b.data = output_b

net2.attention.query_w.data = query_w
net2.attention.key_w.data = key_w
net2.attention.value_w.data = value_w
net2.attention.gating_w.data = gating_w
net2.attention.gating_b.data = gating_b
net2.attention.output_w.data = output_w
net2.attention.output_b.data = output_b


Y1 = net1(q_data, m_data, bias, nonbatched_bias)
Y2 = net2(q_data.type(torch.bfloat16), m_data, bias, nonbatched_bias)
r = Y1.max() - Y1.min()

# print(Y1[3,3,10:15])
# print(Y2[3,3,10:15])
print(
    "    Foward pass check: ",
    ((torch.abs(Y1 - Y2.type(torch.float32)) / r < 0.25).sum() == B * S * HS).item(),
)
# print("diff: ", r)
print(
    " Number of errors: ",
    B * S * HS - (torch.abs(Y1 - Y2.type(torch.float32)) / r < 0.25).sum(),
)


forward1 = 0  # variables to store time values
forward2 = 0

N = 5  # Number of iterations
for _ in range(N):  # MKLDNN PyTorch layer Forward and Backward pass timing
    start = time.time()
    Y1 = net1(q_data, m_data, bias, nonbatched_bias)
    forward1 += time.time() - start

tpp_pytorch_extension.reset_debug_timers()
for _ in range(N):  # Optimized PyTorch layer Forward and Backward pass timing
    start = time.time()
    Y2 = net2(q_data.type(torch.bfloat16), m_data, bias, nonbatched_bias)
    forward2 += time.time() - start
tpp_pytorch_extension.print_debug_timers()

print(
    "Forward pass time (PyTorch layer): {:.3f} ms | Forward pass time (Optimized layer): {:.3f} ms".format(
        forward1 * 1e3 / N, forward2 * 1e3 / N
    )
)

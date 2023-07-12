###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Authors: Ramanarayan Mohanty, Sasikanth Avancha (Intel Corp.)                                     #
###############################################################################

import math
from typing import Callable, Iterable, Tuple
import torch
from torch.optim import Optimizer
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function
from dgl.nn.functional import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
import dgl.function as fn
from dgl.utils import expand_as_pair
from tpp_pytorch_extension.utils.blocked_layout import (
    BlockedParameter,
    BlockedModule,
    BlockedTensor,
)
from tpp_pytorch_extension.utils import blocked_layout, xsmm
from tpp_pytorch_extension._C import _fused_gsage as fused_gat_cpp

# from tpp_pytorch_extension._C import _fused_gat as fused_gat_cpp
import time
from contextlib import contextmanager

import numpy as np


torch.autograd.set_detect_anomaly(False)

USE_BF16_PARAMS = True


class DummyLinear(BlockedModule):
    def __init__(self, in_features, out_features, bias=True):
        super(DummyLinear, self).__init__()
        self.weight = BlockedParameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = BlockedParameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        raise NotImplemented
        return input


class GATMLPAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, align, act, *inputs):

        (mlp_inp, wt, attn, bias) = inputs
        (mlp_out, attn_out, act_mask) = fused_gat_cpp.fused_gat_mlp_attn_fwd(
            align, act, inputs
        )

        if act == "None":
            act_mask = torch.tensor([], dtype=torch.short)
        ctx.save_for_backward(
            mlp_out, attn, mlp_inp, wt, act_mask
        )  # attn_input = mlp_out
        ctx.act = act
        ctx.align = align
        return (attn_out, mlp_out)  # mlp_out = attn_inp

    @staticmethod
    def backward(ctx, *grad_attn_out):
        inputs = list(grad_attn_out)
        inputs += ctx.saved_tensors
        grad_inp, grad_wt, grad_attn, grad_bias = fused_gat_cpp.fused_gat_mlp_attn_bwd(
            ctx.align, ctx.act, inputs
        )

        return (None, None, grad_inp, grad_wt, grad_attn, grad_bias)


class DropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, inp, training):
        outputs = fused_gat_cpp.gat_dropout_fwd(p, inp, training)
        (out, dp_mask) = outputs
        ctx.save_for_backward(dp_mask)
        ctx.p = p
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        # breakpoint()
        grad_inp = fused_gat_cpp.gat_dropout_bwd(ctx.p, inputs)
        return (None, grad_inp, None)


class Dropout_(nn.Module):
    __constants__ = ["p", "inplace"]
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(Dropout_, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return "p={}, inplace={}".format(self.p, self.inplace)


class Dropout(Dropout_):
    r"""Train"""

    def forward(self, input):
        input = input.contiguous()
        output = DropoutFunction.apply(self.p, input, self.training)

        return output


class LeakyReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, input):
        (out, mask) = fused_gat_cpp.leakyrelu_fwd(alpha, input)
        ctx.save_for_backward(input, mask)
        ctx.alpha = alpha

        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp = fused_gat_cpp.leakyrelu_bwd(ctx.alpha, inputs)

        return (None, grad_inp)


class LeakyReLU(nn.Module):
    __constants__ = ["inplace"]

    def __init__(self, alpha, inplace: bool = False):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha
        self.inplace = False  # inplace

    def forward(self, input):
        input = input.contiguous()
        output = LeakyReLUFn.apply(self.alpha, input)
        return output


class GATConvOpt(BlockedModule):
    r"""

    Description
    -----------
    Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    over an input signal.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} &= \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        GATConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    bias : bool, optional
        If True, learns a bias term. Defaults: ``True``.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GATConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> gatconv = GATConv(10, 2, num_heads=3)
    >>> res = gatconv(g, feat)
    >>> res
    tensor([[[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]]], grad_fn=<BinaryReduceBackward>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.bipartite((u, v))
    >>> u_feat = th.tensor(np.random.rand(2, 5).astype(np.float32))
    >>> v_feat = th.tensor(np.random.rand(4, 10).astype(np.float32))
    >>> gatconv = GATConv((5,10), 2, 3)
    >>> res = gatconv(g, (u_feat, v_feat))
    >>> res
    tensor([[[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]],
            [[ 0.0268,  1.0783],
            [ 0.5041, -1.3025],
            [ 0.6568,  0.7048]],
            [[-0.2688,  1.0543],
            [-0.0315, -0.9016],
            [ 0.3943,  0.5347]],
            [[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]]], grad_fn=<BinaryReduceBackward>)
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(GATConvOpt, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._num_heads = num_heads
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.bc = self._in_dst_feats
        self.bk = num_heads * self._out_feats
        self.res = False
        self.align = 32
        self.fdp = feat_drop
        self.adp = attn_drop
        print("Use opt_mlp code---------------")
        for cbf in [50, 32, 16]:
            if self._in_dst_feats % cbf == 0:
                self.bc = cbf
                break

        for kbf in [50, 32, 16]:
            if self._out_feats % kbf == 0:
                self.bk = kbf
                break
        # print("bk: ",self._out_feats, " ", self.bk, " bc ", self.bc)
        if isinstance(in_feats, tuple):
            self.fc_src = DummyLinear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
            self.fc_src.weight.set_blocking_param(
                (
                    [self.bk, self.bc],
                    [0, 2, 3, 1],
                )
            )
            self.fc_dst = DummyLinear(
                self._in_dst_feats, out_feats * num_heads, bias=False
            )
            self.fc_dst.weight.set_blocking_param(
                (
                    [self.bk, self.bc],
                    [0, 2, 3, 1],
                )
            )
        else:
            self.fc = DummyLinear(self._in_src_feats, out_feats * num_heads, bias=False)

            self.fc.weight.set_blocking_param(
                (
                    [self.bk, self.bc],
                    [0, 2, 3, 1],
                )
            )
        ### Optimized-----
        self.attn_l = BlockedParameter(
            torch.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r = BlockedParameter(
            torch.FloatTensor(size=(1, num_heads, out_feats))
        )

        self.feat_drop = Dropout(feat_drop)
        self.attn_drop = Dropout(attn_drop)

        self.leaky_relu = LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer("bias", None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = DummyLinear(
                    self._in_dst_feats, num_heads * out_feats, bias=False
                )
                self.res_fc.weight.set_blocking_param(
                    (
                        [self.bk, self.bc],
                        [0, 2, 3, 1],
                    )
                )
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.activation = (
            "relu" if activation == F.relu else "None"
        )  # For Optimized Block
        self.use_bf16 = False
        self.reset_parameters()

    def maybe_block_params(self):

        if not hasattr(self, "fc_src"):
            self.fc.weight.block()
        else:
            self.fc_src.weight.block()
            self.fc_dst.weight.block()

        if self.res_fc is not None:
            self.res_fc.weight.block()

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]

                if self.fdp > 0.0:
                    h_src = self.feat_drop(feat[0])
                    h_dst = self.feat_drop(feat[1])
                else:
                    h_src = feat[0]
                    h_dst = feat[1]

                if self.use_bf16:
                    h_src = h_src.to(torch.bfloat16)
                    h_dst = h_dst.to(torch.bfloat16)

                if not hasattr(self, "fc_src"):

                    inputs = [h_src, self.fc_src.weight, self.attn_l]
                    if self.use_bf16:
                        inputs = [
                            i.to(torch.bfloat16) if i.is_floating_point() else i
                            for i in inputs
                        ]
                    inputs.append(self.bias)

                    N = h_src.size(0)
                    align = self.align if (N > self.align or N == 0) else N
                    feat_src = feat_dst = GATMLPAttentionFunction.apply(
                        self.align, *inputs
                    ).view(*src_prefix_shape, self._num_heads, self._out_feats)

                else:
                    inputs_src = [
                        h_src,
                        self.fc_src.weight,
                        self.attn_l,
                    ]
                    if self.use_bf16:
                        inputs_src = [
                            i.to(torch.bfloat16) if i.is_floating_point() else i
                            for i in inputs_src
                        ]
                    inputs_src.append(self.bias)
                    N = h_src.size(0)
                    align = self.align if (N > self.align or N == 0) else N

                    el, feat_src_ = GATMLPAttentionFunction.apply(
                        align,
                        self.activation,
                        *inputs_src,
                    )
                    feat_src = feat_src_.view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )

                    inputs_dst = [
                        h_dst,
                        self.fc_dst.weight,
                        self.attn_r,
                    ]
                    if self.use_bf16:
                        inputs_dst = [
                            i.to(torch.bfloat16) if i.is_floating_point() else i
                            for i in inputs_dst
                        ]
                    inputs_dst.append(self.bias)
                    N = h_dst.size(0)
                    align = self.align if (N > self.align or N == 0) else N

                    er, feat_dst_ = GATMLPAttentionFunction.apply(
                        align,
                        self.activation,
                        *inputs_dst,
                    )  #
                    feat_dst = feat_dst_.view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )

            else:

                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = feat

                if graph.is_block:
                    feat_dst = self.feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(),
                    ) + dst_prefix_shape[1:]

            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})

            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


class GATConvOptBF16(GATConvOpt):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):

        super(GATConvOptBF16, self).__init__(
            in_feats,
            out_feats,
            num_heads,
            feat_drop,
            attn_drop,
            negative_slope,
            residual,
            activation,
            allow_zero_in_degree,
            bias,
        )
        if USE_BF16_PARAMS:
            self.fc_src.weight.set_blocking_param(
                (
                    [self.bk, [self.bc // 2, 2]],
                    [0, 2, 3, 1, 4],
                    torch.bfloat16,
                )
            )
            self.fc_dst.weight.set_blocking_param(
                (
                    [self.bk, [self.bc // 2, 2]],
                    [0, 2, 3, 1, 4],
                    torch.bfloat16,
                )
            )

        self.use_bf16 = True

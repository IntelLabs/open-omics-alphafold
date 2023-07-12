###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)                                       #
###############################################################################

import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function
from tpp_pytorch_extension.utils.blocked_layout import (
    BlockedParameter,
    BlockedModule,
    BlockedTensor,
    get_blocking_signature,
)
from tpp_pytorch_extension.utils.xsmm import get_vnni_blocking
from tpp_pytorch_extension._C import _fused_bert_unpad as fused_bert_cpp
import time
from contextlib import contextmanager

from transformers.modeling_utils import apply_chunking_to_forward
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

USE_LOW_PREC_PARAMS = True
LAYER_NORM_USE_FP32_PARAMS = True
global_layer_dtype = torch.float32
print_cou = 0


def print_grad_hook(var, name):
    if not hasattr(var, "grad_fn"):
        return

    def register_grad(grad_input, grad_output):
        global print_cou
        print(f"TESTGRADU {name}: {var.grad_fn.name()} - {grad_input[0].abs().sum()}")
        torch.save(grad_input, "tmp_u_%d.pt" % print_cou)
        print_cou += 1

    var.grad_fn.register_hook(register_grad)


def generate_mask(attention_mask):
    assert not attention_mask is None, "attention_mask is None"
    B, _, _, S = attention_mask.shape
    S1, S2 = BlockedModule.default_blocking_factors(S)
    attention_mask = attention_mask.view([B, S]).clone()
    nnz = (((attention_mask + 10000).count_nonzero(dim=-1) + (S2 - 1)) // S2) * S2
    # nnz = (((attention_mask+10000).count_nonzero(dim=-1) + (S - 1))//S)*S
    nnz1 = nnz.unsqueeze(dim=1).expand([-1, S])
    a = torch.arange(S).expand([B, -1])
    msk = a < nnz1
    attention_mask = attention_mask[msk].clone()
    seq_offsets = torch.cat([torch.zeros([1]), nnz // S2]).to(torch.long)
    seq_sqr_offsets = seq_offsets * seq_offsets
    seq_offsets = seq_offsets.cumsum(dim=0)
    seq_sqr_offsets = seq_sqr_offsets.cumsum(dim=0)
    return msk, attention_mask, seq_offsets, seq_sqr_offsets, S2


class PadInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, msk, padded_shape):
        ctx.save_for_backward(msk)
        output = input.new_zeros(padded_shape)

        output[msk, :] = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (msk,) = ctx.saved_tensors

        grad_input = grad_output[msk, :]
        return grad_input, None, None


class UnpadInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, msk):
        ctx.save_for_backward(msk)
        ctx.shape = input.shape

        output = input[msk, :]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (msk,) = ctx.saved_tensors

        grad_input = grad_output.new_zeros(ctx.shape)
        grad_input[msk, :] = grad_output

        return grad_input, None


# class DummyLinear(BlockedModule):
#     def __init__(self, in_features, out_features, bias=True):
#         super(DummyLinear, self).__init__()
#         self.weight = BlockedParameter(torch.Tensor(out_features, in_features))
#         if bias:
#             self.bias = BlockedParameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter("bias", None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)
#
#     def forward(self, input):
#         raise NotImplemented
#         return input


class DummyLinear(BlockedModule, torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        # super(DummyLinear, self).__init__()
        torch.nn.Linear.__init__(self, in_features, out_features, bias)
        self.weight = BlockedParameter(self.weight.data)
        if bias:
            self.bias = BlockedParameter(self.bias.data)

    def forward(self, input):
        raise NotImplemented
        return input


class DummyLayerNorm(BlockedModule, torch.nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        torch.nn.LayerNorm.__init__(self, *args, **kwargs)
        if self.elementwise_affine:
            self.weight = BlockedParameter(self.weight.data)
            self.bias = BlockedParameter(self.bias.data)

    def forward(self, input):
        raise NotImplemented
        return input


class BertSelfAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, training, need_attention_output, *inputs):
        # print("FWD Called")
        # print("BSAFWD:", [t.shape if isinstance(t, torch.Tensor) else t for t in inputs[6:]])
        (
            context_layer,
            attention_probs_out,
            hs_t,
            ehs_t,
            ql_t,
            kl_tv,
            vl_tv,
            ap,
            apd_t,
            ap_dp_mask,
        ) = fused_bert_cpp.fused_self_attention_fwd(p, inputs, training)
        (qw, qb, kw, kb, vw, vb, hs, am, hm, ehs, eam, offs, offs2) = inputs
        ctx.save_for_backward(
            qw,
            kw,
            vw,
            hs_t,
            hm,
            ehs_t,
            ql_t,
            kl_tv,
            vl_tv,
            ap,
            apd_t,
            ap_dp_mask,
            offs,
            offs2,
        )
        ctx.p = p
        # stop = False
        # for i, t in enumerate([context_layer, attention_probs_out, hs_t, ehs_t, ql_t, kl_tv, vl_tv, ap, apd_t, ap_dp_mask]):
        #    nan = t.isnan().any().item()
        #    stop = stop or nan
        #    if nan: print ("Nan found in %d tensor" % i)
        # if stop: raise "Nan Found"

        # print("Returning from FWD")
        if need_attention_output:
            return context_layer, attention_probs_out
        else:
            return (context_layer,)

    @staticmethod
    def backward(ctx, *grad_outs):
        # print("BWD Called")
        inputs = []
        inputs += [g.contiguous() for g in grad_outs]
        if len(inputs) == 1:
            inputs.append(inputs[0].new_empty(0))
        inputs += ctx.saved_tensors
        p = ctx.p
        (
            dqw,
            dqb,
            dkw,
            dkb,
            dvw,
            dvb,
            dhs,
            dehs,
        ) = fused_bert_cpp.fused_self_attention_bwd(p, inputs)
        ehs = inputs[7]
        if ehs is None:
            dehs = None
        # print("Returning from BWD")
        # print("DHS:", dhs.view([-1])[:4])
        return (
            None,
            None,
            None,
            dqw,
            dqb,
            dkw,
            dkb,
            dvw,
            dvb,
            dhs,
            None,
            None,
            dehs,
            None,
            None,
            None,
        )


class BertSelfAttention(BlockedModule):
    r"""TPP Bert Self Attention Layer using libxsmm blocked GEMM"""

    # __constants__ = ['bias', 'C', 'K']

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        # self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads  # N
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads
        )  # H
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # NH
        self.hidden_size = config.hidden_size  # HS
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

        self.query = DummyLinear(config.hidden_size, self.all_head_size)
        self.key = DummyLinear(config.hidden_size, self.all_head_size)
        self.value = DummyLinear(config.hidden_size, self.all_head_size)
        self.is_decoder = config.is_decoder
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        assert (
            self.position_embedding_type == "absolute"
        ), "self.position_embedding_type other than absolute not supported"

        self.query.weight.set_blocking_param(
            (
                [self.attention_head_size, self.attention_head_size],
                [0, 2, 3, 1],
            )
        )
        self.key.weight.set_blocking_param(
            (
                [self.attention_head_size, self.attention_head_size],
                [0, 2, 3, 1],
            )
        )
        self.value.weight.set_blocking_param(
            (
                [self.attention_head_size, self.attention_head_size],
                [0, 2, 3, 1],
            )
        )
        self.blocked_input_signature = get_blocking_signature("SF", "SFSF")
        layer_dtype = global_layer_dtype
        layer_use_low_prec = layer_dtype != torch.float32
        if layer_use_low_prec == True and USE_LOW_PREC_PARAMS:
            low_prec_vnni_blocking = get_vnni_blocking(layer_dtype)
            self.query.weight.set_blocking_param(
                (
                    [
                        self.attention_head_size,
                        [
                            self.attention_head_size // low_prec_vnni_blocking,
                            low_prec_vnni_blocking,
                        ],
                    ],
                    [0, 2, 3, 1, 4],
                    layer_dtype,
                )
            )
            self.key.weight.set_blocking_param(
                (
                    [
                        self.attention_head_size,
                        [
                            self.attention_head_size // low_prec_vnni_blocking,
                            low_prec_vnni_blocking,
                        ],
                    ],
                    [0, 2, 3, 1, 4],
                    layer_dtype,
                )
            )
            self.value.weight.set_blocking_param(
                (
                    [
                        self.attention_head_size,
                        [
                            self.attention_head_size // low_prec_vnni_blocking,
                            low_prec_vnni_blocking,
                        ],
                    ],
                    [0, 2, 3, 1, 4],
                    layer_dtype,
                )
            )
            self.query.bias.set_blocking_param((None, None, layer_dtype))
            self.key.bias.set_blocking_param((None, None, layer_dtype))
            self.value.bias.set_blocking_param((None, None, layer_dtype))
        self.layer_dtype = layer_dtype

        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def maybe_block_params(self):
        self.query.weight.block()
        self.key.weight.block()
        self.value.weight.block()
        self.query.bias.block()
        self.key.bias.block()
        self.value.bias.block()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        seq_offsets=None,
        seq_sqr_offsets=None,
        S2=None,
    ):
        assert past_key_value == None, "past_key_value not supported"
        self.maybe_block_params()
        if encoder_hidden_states is not None:
            assert (
                encoder_hidden_states.shape == hidden_states.shape
            ), "Different shapes not supported(%s != %s)" % (
                encoder_hidden_states.shape,
                hidden_states.shape,
            )
            encoder_hidden_states = self.get_blocked_tensor(
                encoder_hidden_states,
                self.blocked_input_signature,
                [S2, self.attention_head_size],
            )
        orig_hidden_states = hidden_states
        hidden_states = self.get_blocked_tensor(
            hidden_states,
            self.blocked_input_signature,
            [S2, self.attention_head_size],
        )
        hidden_states = hidden_states.cvt_to(self.layer_dtype)
        # print(f"hidden_states: {hidden_states.shape}")
        inputs = [
            self.query.weight,
            self.query.bias,
            self.key.weight,
            self.key.bias,
            self.value.weight,
            self.value.bias,
        ]
        inputs.append(hidden_states)
        if attention_mask is not None:
            # print(f"attention_mask: {attention_mask.shape}")
            # B, S1, N, S2, H = hidden_states.shape
            # S = S1 * S2
            # print("Before attention_mask shape = %s (%s)" % (attention_mask.shape, attention_mask.numel()))
            # attention_mask = attention_mask.expand([B, N, S, S]).view([B, N, S1, S2, S1, S2]).permute([0, 2, 1, 4, 3, 5]).contiguous()
            # assert (
            #     attention_mask.size(1) == attention_mask.size(2) == 1
            # ), "unsupported attention_mask shape %s" % (attention_mask.shape,)
            attention_mask = attention_mask.contiguous()
            # print("After  attention_mask shape = %s (%s)" % (attention_mask.shape, attention_mask.numel()))
        if head_mask is not None:
            print(f"head_mask: {head_mask.shape}")
        if encoder_attention_mask is not None:
            print(f"encoder_attention_mask: {encoder_attention_mask.shape}")
            # B, S1, N, S2, H = encoder_hidden_states.shape
            # S = S1 * S2
            # encoder_attention_mask = encoder_attention_mask.expand([B, N, S, S]).view([B, N, S1, S2, S1, S2]).permute([0, 2, 1, 4, 3, 5]).contiguous()
            assert (
                encoder_attention_mask.size(1) == encoder_attention_mask.size(2) == 1
            ), "unsupported encoder_attention_mask shape %s" % (
                encoder_attention_mask.shape,
            )
            encoder_attention_mask = encoder_attention_mask.contiguous()
        inputs.append(attention_mask if attention_mask is not None else torch.Tensor())
        inputs.append(head_mask if head_mask is not None else torch.Tensor())
        inputs.append(
            encoder_hidden_states
            if encoder_hidden_states is not None
            else torch.Tensor()
        )
        inputs.append(
            encoder_attention_mask
            if encoder_attention_mask is not None
            else torch.Tensor()
        )
        inputs.append(seq_offsets if seq_offsets is not None else torch.Tensor())
        inputs.append(
            seq_sqr_offsets if seq_sqr_offsets is not None else torch.Tensor()
        )

        # context_layer, attention_probs = fused_bert_cpp.forward(self.handle.handle, inputs)
        p = self.attention_probs_dropout_prob if self.training else 0.0
        inputs = [
            i.cvt_to(self.layer_dtype) if i.is_floating_point() else i for i in inputs
        ]
        outputs = BertSelfAttentionFunction.apply(
            p, self.training, output_attentions, *inputs
        )
        # outputs = BertSelfAttentionFunction.apply(p, self.training, True, *inputs)
        context_layer = outputs[0]

        context_layer = BlockedTensor(
            context_layer, self.blocked_input_signature, orig_hidden_states.dtype
        )
        if output_attentions:
            print("Reshaping output_attentions")
            attention_probs = outputs[1]
            attention_probs = (
                attention_probs.permute([0, 2, 1, 4, 3, 5])
                .contiguous()
                .view([B, self.num_attention_heads, S, S])
                .cvt_to(orig_hidden_states.dtype)
            )

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


class BertOutputBaseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, eps, training, *inputs):
        (inp, inp2, wt, bias, gamma, beta) = inputs
        # print("A")
        outputs = fused_bert_cpp.fused_dense_dropout_layernorm_fwd(
            p, eps, inputs, training
        )
        # print("B")
        (out, dout, mean, var, dp_mask) = outputs
        ctx.save_for_backward(inp, wt, gamma, mean, var, dout, dp_mask)
        # print("C")
        ctx.p = p
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        (
            grad_inp,
            grad_inp2,
            grad_wt,
            grad_bias,
            grad_gamma,
            grad_beta,
        ) = fused_bert_cpp.fused_dense_dropout_layernorm_bwd(ctx.p, inputs)
        return (
            None,
            None,
            None,
            grad_inp,
            grad_inp2,
            grad_wt,
            grad_bias,
            grad_gamma,
            grad_beta,
        )


class BertOutputBase(BlockedModule):
    def __init__(self, config, selfOutput):
        super().__init__()
        ifm = config.hidden_size if selfOutput else config.intermediate_size
        self.dense = DummyLinear(ifm, config.hidden_size)
        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm = DummyLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.layer_norm_eps = config.layer_norm_eps
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense.weight.set_blocking_param(
            (
                [self.attention_head_size, self.attention_head_size],
                [0, 2, 3, 1],
            )
        )
        self.blocked_input_signature = get_blocking_signature("SF", "SFSF")
        layer_dtype = global_layer_dtype
        layer_use_low_prec = layer_dtype != torch.float32
        if layer_use_low_prec == True and USE_LOW_PREC_PARAMS:
            low_prec_vnni_blocking = get_vnni_blocking(layer_dtype)
            self.dense.weight.set_blocking_param(
                (
                    [
                        self.attention_head_size,
                        [
                            self.attention_head_size // low_prec_vnni_blocking,
                            low_prec_vnni_blocking,
                        ],
                    ],
                    [0, 2, 3, 1, 4],
                    layer_dtype,
                )
            )
            self.dense.bias.set_blocking_param((None, None, layer_dtype))
            if not LAYER_NORM_USE_FP32_PARAMS:
                self.LayerNorm.weight.set_blocking_param((None, None, layer_dtype))
                self.LayerNorm.bias.set_blocking_param((None, None, layer_dtype))
        self.layer_dtype = layer_dtype
        # print(f"config.hidden_size = {config.hidden_size}, ifm = {ifm}, p = {config.hidden_dropout_prob}, eps = {config.layer_norm_eps}")

    def maybe_block_params(self):
        self.dense.weight.block()
        self.dense.bias.block()
        self.LayerNorm.weight.block()
        self.LayerNorm.bias.block()

    def forward(self, hidden_states, input_tensor):
        self.maybe_block_params()
        orig_hidden_states = hidden_states
        hidden_states = self.get_blocked_tensor(
            hidden_states,
            self.blocked_input_signature,
            [None, self.attention_head_size],
        )
        hidden_states = hidden_states.cvt_to(self.layer_dtype)
        input_tensor = self.get_blocked_tensor(
            input_tensor,
            self.blocked_input_signature,
            [None, self.attention_head_size],
        )
        input_tensor = input_tensor.cvt_to(self.layer_dtype)

        inputs = [
            hidden_states,
            input_tensor,
            self.dense.weight,
            self.dense.bias,
        ]
        if not LAYER_NORM_USE_FP32_PARAMS:
            inputs += [self.LayerNorm.weight, self.LayerNorm.bias]
        p = self.hidden_dropout_prob if self.training else 0.0
        inputs = [
            i.cvt_to(self.layer_dtype) if i.is_floating_point() else i for i in inputs
        ]
        if LAYER_NORM_USE_FP32_PARAMS:
            inputs += [self.LayerNorm.weight, self.LayerNorm.bias]
        ret = BertOutputBaseFunction.apply(
            p, self.layer_norm_eps, self.training, *inputs
        )
        # ret = ret.cvt_to(hidden_states.dtype)
        ret = BlockedTensor(ret, self.blocked_input_signature, orig_hidden_states.dtype)
        return ret
        # hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # return hidden_states


class BertSelfOutput(BertOutputBase):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__(config, True)


class BertOutput(BertOutputBase):
    def __init__(self, config):
        super(BertOutput, self).__init__(config, False)


class BertIntermediateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, act, training):
        # assert act == "gelu_new", "%s activation type is not supported" % act
        gelu_in, output = fused_bert_cpp.fused_dense_gelu_fwd(
            input, weight, bias, training
        )
        ctx.save_for_backward(input, weight, gelu_in)
        ctx.act = act
        return output

    @staticmethod
    def backward(ctx, grad_out):
        (input, weight, gelu_in) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_inp, grad_wt, grad_bias = fused_bert_cpp.fused_dense_gelu_bwd(
            grad_out, gelu_in, input, weight
        )
        return (grad_inp, grad_wt, grad_bias, None, None)


class BertIntermediate(BlockedModule):
    def __init__(self, config):
        super().__init__()
        self.dense = DummyLinear(config.hidden_size, config.intermediate_size)
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.dense.weight.set_blocking_param(
            (
                [self.attention_head_size, self.attention_head_size],
                [0, 2, 3, 1],
            )
        )
        assert config.hidden_act in ["gelu", "gelu_new"], (
            "Currently, only GELU new is supported in fused op, %s is given"
            % config.hidden_act
        )
        self.hidden_act = config.hidden_act
        self.blocked_input_signature = get_blocking_signature("SF", "SFSF")
        layer_dtype = global_layer_dtype
        layer_use_low_prec = layer_dtype != torch.float32
        if layer_use_low_prec == True and USE_LOW_PREC_PARAMS:
            low_prec_vnni_blocking = get_vnni_blocking(layer_dtype)
            self.dense.weight.set_blocking_param(
                (
                    [
                        self.attention_head_size,
                        [
                            self.attention_head_size // low_prec_vnni_blocking,
                            low_prec_vnni_blocking,
                        ],
                    ],
                    [0, 2, 3, 1, 4],
                    layer_dtype,
                )
            )
            self.dense.bias.set_blocking_param((None, None, layer_dtype))

        self.layer_dtype = layer_dtype
        # if isinstance(config.hidden_act, str):
        #     self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # else:
        #     self.intermediate_act_fn = config.hidden_act

    def maybe_block_params(self):
        self.dense.weight.block()
        self.dense.bias.block()

    def forward(self, hidden_states):
        self.maybe_block_params()
        orig_hidden_states = hidden_states
        hidden_states = self.get_blocked_tensor(
            hidden_states,
            self.blocked_input_signature,
            [None, self.attention_head_size],
        )
        hidden_states = hidden_states.cvt_to(self.layer_dtype)
        inputs = [hidden_states, self.dense.weight, self.dense.bias]
        inputs = [
            i.cvt_to(self.layer_dtype) if i.is_floating_point() else i for i in inputs
        ]
        ret = BertIntermediateFunction.apply(*inputs, self.hidden_act, self.training)
        # ret = ret.cvt_to(hidden_states.dtype)
        hidden_states = BlockedTensor(
            ret, self.blocked_input_signature, orig_hidden_states.dtype
        )
        # hidden_states = self.dense(hidden_states)
        # hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertEmbeddingsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, training, prob, eps, head_size, pad_id, *inputs):
        (ii, pi, ti, ie, g, b, we, pe, te) = inputs
        (
            out,
            eout,
            mean,
            var,
            msk,
        ) = fused_bert_cpp.fused_embedding_layernorm_dropout_fwd(
            prob, eps, head_size, pad_id, inputs, training
        )
        ctx.save_for_backward(ii, pi, ti, ie, g, we, pe, te, mean, var, eout, msk)
        ctx.prob = prob
        ctx.pad_id = pad_id
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        prob = ctx.prob
        pad_id = ctx.pad_id
        inputs = []
        inputs += [t.contiguous() for t in grad_outs]
        inputs += ctx.saved_tensors
        (
            die,
            dg,
            db,
            dwe,
            dpe,
            dte,
        ) = fused_bert_cpp.fused_embedding_layernorm_dropout_bwd(prob, pad_id, inputs)
        grad_inps = (
            None,
            None,
            None,
            die,
            dg,
            db,
            dwe,
            dpe,
            dte,
        )
        return (None, None, None, None, None) + grad_inps


class BertEmbeddings(BlockedModule):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = DummyLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm_eps = config.layer_norm_eps
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.pad_token_id = config.pad_token_id

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        assert (
            self.position_embedding_type == "absolute"
        ), f"position embedding type {self.position_embedding_type} not supported"
        self.blocked_ids_signature = get_blocking_signature("BS", "BSS")
        self.blocked_embed_signature = get_blocking_signature("BSF", "BSFSF")
        layer_dtype = global_layer_dtype
        layer_use_low_prec = layer_dtype != torch.float32
        # We use bfloat16 for embedding layer even when other layers use bfloat8
        self.layer_dtype = layer_dtype
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(
                f"config.hidden_size = {config.hidden_size}, config.intermediate_size = {config.intermediate_size}, p = {config.hidden_dropout_prob}, eps = {config.layer_norm_eps}, prec = {global_layer_dtype}"
            )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        assert past_key_values_length == 0, "past_key_values_length != 0 Not supported"
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = self.get_blocked_tensor(
                input_ids, self.blocked_ids_signature, [None, None]
            )
        else:
            input_shape = inputs_embeds.size()[:-1]
            input_ids = torch.LongTensor()
            inputs_embeds = self.get_blocked_tensor(
                inputs_embeds,
                self.blocked_embed_signature,
                [None, self.attention_head_size],
            )

        # seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.LongTensor()
        else:
            position_ids = self.get_blocked_tensor(
                position_ids, self.blocked_ids_signature, [None, None]
            )

        if token_type_ids is None:
            token_type_ids = torch.LongTensor()
        else:
            token_type_ids = self.get_blocked_tensor(
                token_type_ids, self.blocked_ids_signature, [None, None]
            )

        if inputs_embeds is None:
            inputs_embeds = torch.Tensor()
        #     inputs_embeds = self.word_embeddings(input_ids)
        # position_embeddings = self.position_embeddings(position_ids)
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        emb_weighs = [
            self.word_embeddings.weight,
            self.position_embeddings.weight,
            self.token_type_embeddings.weight,
        ]
        inputs = [
            input_ids,
            position_ids,
            token_type_ids,
            inputs_embeds,
            self.LayerNorm.weight,
            self.LayerNorm.bias,
        ]
        p = self.hidden_dropout_prob if self.training else 0.0
        if not LAYER_NORM_USE_FP32_PARAMS:
            inputs = [
                i.cvt_to(self.layer_dtype) if i.is_floating_point() else i
                for i in inputs
            ]
        # inputs += [e.cvt_to(torch.bfloat8) for e in emb_weighs]
        inputs += emb_weighs
        embeddings = BertEmbeddingsFunction.apply(
            self.training,
            p,
            self.layer_norm_eps,
            self.attention_head_size,
            self.pad_token_id,
            *inputs,
        )
        embeddings = BlockedTensor(
            embeddings, self.blocked_embed_signature, torch.float32
        )
        # embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)
        return embeddings


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        seq_offsets=None,
        seq_sqr_offsets=None,
        S2=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            seq_offsets=seq_offsets,
            seq_sqr_offsets=seq_sqr_offsets,
            S2=S2,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert (
                self.is_decoder
            ), f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        seq_offsets=None,
        seq_sqr_offsets=None,
        S2=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            seq_offsets=seq_offsets,
            seq_sqr_offsets=seq_sqr_offsets,
            S2=S2,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
                seq_offsets=seq_offsets,
                seq_sqr_offsets=seq_sqr_offsets,
                S2=S2,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        # self.blocked_input_signature = get_blocking_signature(
        #    "SF", "SFSF"
        # )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None
        if hasattr(hidden_states, "unblocked_tensor"):
            hidden_states = hidden_states.unblocked_tensor()
        padded_shape = hidden_states.shape
        # print_grad_hook(hidden_states, 'BertEncoder:hidden_states')
        msk, attention_mask, seq_offsets, seq_sqr_offsets, S2 = generate_mask(
            attention_mask
        )
        hidden_states = UnpadInput.apply(hidden_states, msk)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    seq_offsets=seq_offsets,
                    seq_sqr_offsets=seq_sqr_offsets,
                    S2=S2,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    seq_offsets=seq_offsets,
                    seq_sqr_offsets=seq_sqr_offsets,
                    S2=S2,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if hasattr(hidden_states, "unblocked_tensor"):
            hidden_states = hidden_states.unblocked_tensor()
        hidden_states = PadInput.apply(hidden_states, msk, padded_shape)
        # print_grad_hook(hidden_states, 'BertEncoder:hidden_states')

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = DummyLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# bm_default_blocking_factors = BlockedModule.default_blocking_factors
# @staticmethod
# def custom_blocking_factors(S):
#     print(f"S = {S}")
#     if S % 32 == 0: return [S//32, 32]
#     return bm_default_blocking_factors
# BlockedModule.default_blocking_factors = custom_blocking_factors

try:
    import transformers

    transformers_orig_is_tensor = transformers.file_utils.is_tensor

    def is_tensor(x):
        """Tests if ``x`` is a :obj:`torch.Tensor`, :obj:`tf.Tensor` or :obj:`np.ndarray`."""
        if transformers_orig_is_tensor(x):
            return True
        if isinstance(x, BlockedTensor):
            return True
        return False

    transformers.file_utils.is_tensor = is_tensor
except:
    pass


@contextmanager
def tpp_impl(enable=True, use_low_prec=False, use_bf8=False):
    global global_layer_dtype
    try:
        import transformers

        orig_BertEncoder = transformers.models.bert.modeling_bert.BertEncoder
        orig_BertEmbeddings = transformers.models.bert.modeling_bert.BertEmbeddings
        orig_global_layer_dtype = global_layer_dtype
        try:
            if enable:
                transformers.models.bert.modeling_bert.BertEncoder = BertEncoder
                transformers.models.bert.modeling_bert.BertEmbeddings = BertEmbeddings
                if use_low_prec:
                    global_layer_dtype = (
                        torch.bfloat8 if use_bf8 == True else torch.bfloat16
                    )
            yield
        finally:
            transformers.models.bert.modeling_bert.BertEncoder = orig_BertEncoder
            transformers.models.bert.modeling_bert.BertEmbeddings = orig_BertEmbeddings
            global_layer_dtype = orig_global_layer_dtype
    except ImportError as e:
        pass


def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()

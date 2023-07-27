import math

import torch
import torch.nn as nn
from fairseq.modules.fairseq_dropout import FairseqDropout

from ...data_structures.four_dim_pocket import FourDimPocket


class SumThenReg3RpeSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,

        num_summary,

        rel_embeddings,

        layer_idx,

        dropout=0.0,  # attention dropout

        query_proj_bias=True,
        key_proj_bias=True,
        value_proj_bias=True,
        out_proj_bias=True,

        no_rel_proj=None,
        rel_proj_bias=True,

        share_reg_kv_proj=False,

        max_summary=None,

        single_head_masks=False,

        **kwargs
    ):
        assert single_head_masks, "Currently, we only support single head masks."

        super().__init__()
        self.layer_idx = layer_idx
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.single_head_masks = single_head_masks
        self.num_summary = num_summary
        self.max_summary = self.num_summary if max_summary is None else max_summary

        self.pocket = FourDimPocket()
        self.instant_pocket = self.pocket['instant']
        constant_pocket = self.pocket['constant']
        layer_to_sv = constant_pocket['layer_to_sv']
        self.layer_sv = layer_to_sv[self.layer_idx]

        self.share_reg_kv_proj = share_reg_kv_proj

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "attention_embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.num_relation_types = len(rel_embeddings)
        assert self.num_relation_types == 0, "relative position is not open"
        self.rel_embeddings = rel_embeddings
        self.num_rel_embeddings = [item.shape[0] for item in self.rel_embeddings]
        self.rel_dims = [item.shape[1] for item in self.rel_embeddings]

        if self.num_summary > 0:
            self.embed_sum = nn.Embedding(self.max_summary + 1, self.embed_dim, padding_idx=0)
            self.sum_gate = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            self.reg_gate = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            self.gate_bias = nn.Parameter(torch.randn(self.embed_dim), requires_grad=True)

            self.reg_query1_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=query_proj_bias)
        self.reg_query2_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=query_proj_bias)
        self.reg_key_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=key_proj_bias)
        if self.share_reg_kv_proj:
            self.reg_value_proj = None
        else:
            self.reg_value_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=value_proj_bias)

        for idx in range(self.num_relation_types):
            if no_rel_proj is not None and no_rel_proj[idx]:
                key_rel_proj = None
            else:
                rel_dim = self.rel_dims[idx]
                key_rel_proj = nn.Linear(rel_dim, embed_dim, bias=rel_proj_bias)
            setattr(self, 'rel%d_proj' % idx, key_rel_proj)

        self.reg_out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=out_proj_bias)

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'reg_query1_proj'):
            nn.init.xavier_uniform_(self.reg_query1_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.reg_query2_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.reg_key_proj.weight, gain=1 / math.sqrt(2))
        if not self.share_reg_kv_proj:
            nn.init.xavier_uniform_(self.reg_value_proj.weight, gain=1 / math.sqrt(2))

        nn.init.xavier_uniform_(self.reg_out_proj.weight)
        if self.reg_out_proj.bias is not None:
            nn.init.constant_(self.reg_out_proj.bias, 0.0)

        for idx in range(self.num_relation_types):
            rel_proj = getattr(self, 'rel%d_proj' % idx, None)
            if rel_proj is not None:
                nn.init.xavier_uniform_(rel_proj.weight)

    def forward(
        self,
        x: tuple,  # (sum_len, bsz, embed_dim), (reg_len, bsz, embed_dim)
        sum_token_ids,  # (bsz, sum_len)
        sum_len,
        reg_len,
        rel_indices,  # relation list of dict of parts
        key_padding_mask=None,  # (bsz, all_seq_len)
        attn_mask=None,
        need_weights: bool = False,
        need_head_weights: bool = False,
        *args, **kwargs,
    ):
        if key_padding_mask is not None:
            raise NotImplementedError("Please combine key_padding_mask into attn_mask ahead.")
        del key_padding_mask

        if need_head_weights:
            need_weights = True

        # ===== Input Checking =====
        sum_x, reg_x = x
        bsz = reg_x.shape[1]
        assert sum_token_ids is None or sum_token_ids.shape == (bsz, sum_len)
        assert len(rel_indices) == self.num_relation_types

        # ===== Rel Indices =====
        r_list, rel_indices = self.select_and_do_r_proj(rel_indices)
        # r_list: relation list of dict of real_parts (num_selected_rel, embed_dim)
        # rel_indices_for_sum: relation list of dict of real_parts

        # ===== Summarize =====
        base_reg_k = self.reg_key_proj(reg_x)
        base_reg_k = base_reg_k.view(reg_len, bsz, self.num_heads, self.head_dim)
        # base_reg_k: (reg_len, bsz, num_heads, head_dim)

        if self.share_reg_kv_proj:
            base_reg_v = base_reg_k
        else:
            base_reg_v = self.reg_value_proj(reg_x)
            base_reg_v = base_reg_v.view(reg_len, bsz, self.num_heads, self.head_dim)
            # base_reg_v: (reg_len, bsz, num_heads, head_dim)

        if sum_len > 0:
            sum_x = self.embed_sum(sum_token_ids.transpose(0, 1))  # (sum_len, bsz, embed_dim)
            base_sum_q = sum_x.view(sum_len, bsz, self.num_heads, self.head_dim)
            rel_sum_qs = [base_sum_q for _ in range(self.num_relation_types)]
            # base_sum_q: (sum_len, bsz, num_heads, head_dim)  rel_sum_qs: list of (sum_len, bsz, num_heads, head_dim)

            attn_scores_inc_sr = self.do_qk_scores_for_sr(
                base_sum_q, base_reg_k,
                bsz, sum_len, reg_len,
                attn_mask=attn_mask
            )  # real_parts dict of sample list of (heads, head_selected_blocks, block, block)

            attn_scores_inc_sr = self.add_rpe_for_sr(attn_scores_inc_sr, rel_sum_qs, r_list, rel_indices,
                                                     bsz, sum_len, attn_mask=attn_mask)

            attn_scores_inc_sr = self.do_masking_for_sr(attn_scores_inc_sr, attn_mask)

            attn_weights_inc_sr = self.do_attn_softmax_for_sr(attn_scores_inc_sr, attn_mask=attn_mask)
            del attn_scores_inc_sr

            sum_x2 = self.do_av_mul_for_sr(
                attn_weights_inc_sr, base_reg_v, attn_mask=attn_mask, tgt_len=sum_len
            )  # samples list of (sum_len, 1, num_heads, head_dim)
            assert sum_x2.shape == (sum_len, bsz, self.num_heads, self.head_dim)

            base_sum_k2 = sum_x2  # (sum_len, bsz, num_heads, head_dim)
            base_sum_v2 = sum_x2  # (sum_len, bsz, num_heads, head_dim)

        else:
            base_sum_k2 = None
            base_sum_v2 = None

        # ===== Updating =====
        # ----- rs -----
        if sum_len > 0:
            base_reg_q1 = self.reg_query1_proj(reg_x)
            base_reg_q1 = base_reg_q1.view(reg_len, bsz, self.num_heads, self.head_dim)
            rel_reg_qs1 = [base_reg_q1 for _ in range(self.num_relation_types)]

            attn_scores_inc_rs = self.do_qk_scores_for_rs(
                base_reg_q1, base_sum_k2,
                bsz, sum_len, reg_len,
                attn_mask=attn_mask
            )

            attn_scores_inc_rs = self.add_rpe_for_rs(attn_scores_inc_rs, rel_reg_qs1, r_list, rel_indices,
                                                     bsz, reg_len, attn_mask=attn_mask)

            attn_scores_inc_rs = self.do_masking_for_rs(attn_scores_inc_rs, attn_mask)

            attn_weights_inc_rs = self.do_attn_softmax_for_rs(attn_scores_inc_rs, attn_mask=attn_mask)

            sum_output = self.do_av_mul_for_rs(
                attn_weights_inc_rs, base_sum_v2, attn_mask=attn_mask, tgt_len=reg_len,
            )  # (reg_len, bsz, num_heads, head_dim)
        else:
            sum_output = None

        # ----- rr -----
        base_reg_q2 = self.reg_query2_proj(reg_x)
        base_reg_q2 = base_reg_q2.view(reg_len, bsz, self.num_heads, self.head_dim)
        rel_reg_qs2 = [base_reg_q2 for _ in range(self.num_relation_types)]

        attn_scores_inc_rr = self.do_qk_scores_for_rr(
            base_reg_q2, base_reg_k,
            bsz, reg_len, attn_mask=attn_mask,
        )

        attn_scores_inc_rr = self.add_rpe_for_rr(attn_scores_inc_rr, rel_reg_qs2, r_list, rel_indices,
                                                 bsz, reg_len, attn_mask=attn_mask)

        attn_scores_inc_rr = self.do_masking_for_rr(attn_scores_inc_rr, attn_mask)

        attn_weights_inc_rr = self.do_attn_softmax_for_rr(attn_scores_inc_rr, attn_mask=attn_mask)

        reg_output = self.do_av_mul_for_rr(
            attn_weights_inc_rr, base_reg_v, attn_mask=attn_mask, tgt_len=reg_len
        )  # (reg_len, bsz, num_heads, head_dim)

        # ----- gate to combine sum_output and reg_output -----
        reg_output = reg_output.view(reg_len, bsz, self.embed_dim)
        if self.num_summary > 0:
            if sum_output is None:
                sum_output = reg_x.new_zeros(reg_len, bsz, self.embed_dim)
            else:
                sum_output = sum_output.view(reg_len, bsz, self.embed_dim)
            gate = self.sum_gate(sum_output) + self.reg_gate(reg_output) + self.gate_bias
            gate = torch.sigmoid(gate)
            attn_output = gate * sum_output + (1 - gate) * reg_output
        else:
            attn_output = reg_output

        attn_output = self.reg_out_proj(attn_output)

        if need_weights:
            raise NotImplementedError
        else:
            attn_weights = None

        return (None, attn_output), attn_weights
        # (sum_len, bsz, embed_dim)  (reg_len, bsz, embed_dim)
        # None, (bsz, num_heads, all_seq_len, all_seq_len) or (bsz, all_seq_len, all_seq_len)

    def do_qk_scores_for_sr(self, base_sum_q, base_reg_k, bsz, sum_len, reg_len, **kwargs):
        raise NotImplementedError

    def select_and_do_r_proj(self, rel_indices):
        raise NotImplementedError

    def add_rpe_for_sr(self, attn_scores_for_sum, rel_sum_qs, r_list, rel_indices, bsz, sum_len, **kwargs):
        raise NotImplementedError

    def do_masking_for_sr(self, attn_scores_for_sum, attn_mask):
        raise NotImplementedError

    def do_attn_softmax_for_sr(self, attn_scores_for_sum, **kwargs):
        raise NotImplementedError

    def do_av_mul_for_sr(self, attn_weights_inc_sr, base_reg_v, **kwargs):
        raise NotImplementedError

    def do_qk_scores_for_rs(
        self,
        reg_q, sum_k,
        bsz, sum_len, reg_len,
        **kwargs
    ):
        raise NotImplementedError

    def add_rpe_for_rs(self, attn_scores_inc_rs, rel_reg_qs, r_list, rel_indices,
                       bsz, reg_len, **kwargs):
        raise NotImplementedError

    def do_masking_for_rs(self, attn_scores_for_reg, attn_mask):
        raise NotImplementedError

    def do_attn_softmax_for_rs(self, attn_scores_for_reg, attn_mask=None):
        raise NotImplementedError

    def do_av_mul_for_rs(self, attn_weights_for_reg, base_reg_v, **kwargs):
        raise NotImplementedError

    def do_qk_scores_for_rr(
        self,
        reg_q, sum_k,
        bsz, reg_len,
        **kwargs
    ):
        raise NotImplementedError

    def add_rpe_for_rr(self, attn_scores_inc_rs, rel_reg_qs, r_list, rel_indices,
                       bsz, reg_len, **kwargs):
        raise NotImplementedError

    def do_masking_for_rr(self, attn_scores_for_reg, attn_mask):
        raise NotImplementedError

    def do_attn_softmax_for_rr(self, attn_scores_for_reg, attn_mask=None):
        raise NotImplementedError

    def do_av_mul_for_rr(self, attn_weights_for_reg, base_reg_v, **kwargs):
        raise NotImplementedError

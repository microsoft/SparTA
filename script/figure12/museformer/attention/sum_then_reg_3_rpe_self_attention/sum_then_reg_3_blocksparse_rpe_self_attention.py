import torch

from .sum_then_reg_3_rpe_self_attention import SumThenReg3RpeSelfAttention
from ..common.blocksparse_common_operations.qk_mul.qk_mul_1 import do_qk_scores_for_part
from ..common.blocksparse_common_operations.softmax.softmax_1 import do_attn_softmax_for_part
from ..common.blocksparse_common_operations.av_mul.av_mul_1 import do_av_mul_for_part
from ..common.blocksparse_common_operations.rpe.rpe_1 import add_rpe_for_part
from ..common.blocksparse_common_operations.rel_proj.rel_proj_1 import select_and_do_r_proj


class SumThenRegBlocksparse3RpeSelfAttention(SumThenReg3RpeSelfAttention):
    def __init__(self, *args, block_size=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size

        # --- for indexing relative position embeddings ---
        num_heads_arange = torch.arange(self.num_heads)[:, None, None, None]  # (num_heads, 1, 1, 1)
        self.register_buffer('num_heads_arange', num_heads_arange, persistent=False)
        block_size_arange = torch.arange(self.block_size)[None, None, :, None]  # (1, 1, block_size, 1)
        self.register_buffer('block_size_arange', block_size_arange, persistent=False)

    # ============= Interfaces =============
    def select_and_do_r_proj(self, rel_indices):
        """

        :param rel_indices: relation list of real_part dict
        :return:
        """
        return select_and_do_r_proj(self, rel_indices)

    def do_qk_scores_for_sr(self, base_sum_q, base_reg_k, bsz, sum_len, reg_len, attn_mask=None):
        # base_sum_q: (sum_len, bsz, heads, head_dim)
        # base_reg_k: (reg_len, bsz, heads, head_dim)
        return do_qk_scores_for_part(self, base_sum_q, base_reg_k, bsz, sum_len, reg_len, attn_mask, 'sr')

    def add_rpe_for_sr(self, attn_scores_inc_sr, rel_sum_qs, r_list, rel_indices, bsz, sum_len, attn_mask=None):
        return add_rpe_for_part(self, attn_scores_inc_sr, rel_sum_qs, r_list, rel_indices,
                                bsz, sum_len, attn_mask, 'sr')

    def do_masking_for_sr(self, attn_scores_inc_sr, attn_mask):
        return attn_scores_inc_sr

    def do_attn_softmax_for_sr(self, attn_scores_inc_sr, attn_mask=None):
        return do_attn_softmax_for_part(self, attn_scores_inc_sr, attn_mask, 'sr')

    def do_av_mul_for_sr(self, attn_weights_inc_sr, base_reg_v, attn_mask=None, tgt_len=None):
        return do_av_mul_for_part(self, attn_weights_inc_sr, base_reg_v, attn_mask, 'sr', tgt_len)

    def do_qk_scores_for_rs(
        self,
        reg_q, sum_k,
        bsz, sum_len, reg_len,
        attn_mask=None
    ):
        return do_qk_scores_for_part(self, reg_q, sum_k, bsz, reg_len, sum_len, attn_mask, 'rs')

    def add_rpe_for_rs(self, attn_scores_inc_rs, rel_reg_qs, r_list, rel_indices,
                       bsz, reg_len, attn_mask=None):
        return add_rpe_for_part(self, attn_scores_inc_rs, rel_reg_qs, r_list, rel_indices,
                                bsz, reg_len, attn_mask, 'rs')

    def do_masking_for_rs(self, attn_scores_for_reg, attn_mask):
        return attn_scores_for_reg

    def do_attn_softmax_for_rs(self, attn_scores_inc_rs, attn_mask=None):
        return do_attn_softmax_for_part(self, attn_scores_inc_rs, attn_mask, 'rs')

    def do_av_mul_for_rs(self, attn_weights_for_reg, base_reg_v, attn_mask=None, tgt_len=None):
        return do_av_mul_for_part(self, attn_weights_for_reg, base_reg_v, attn_mask, 'rs', tgt_len)

    def do_qk_scores_for_rr(
        self,
        reg_q, sum_k,
        bsz, reg_len,
        attn_mask=None
    ):
        return do_qk_scores_for_part(self, reg_q, sum_k, bsz, reg_len, reg_len, attn_mask, 'rr')

    def add_rpe_for_rr(self, attn_scores_inc_rr, rel_reg_qs, r_list, rel_indices,
                       bsz, reg_len, attn_mask=None):
        return add_rpe_for_part(self, attn_scores_inc_rr, rel_reg_qs, r_list, rel_indices,
                                bsz, reg_len, attn_mask, 'rr')

    def do_masking_for_rr(self, attn_scores_for_rr, attn_mask):
        return attn_scores_for_rr

    def do_attn_softmax_for_rr(self, attn_scores_inc_rr, attn_mask=None):
        return do_attn_softmax_for_part(self, attn_scores_inc_rr, attn_mask, 'rr')

    def do_av_mul_for_rr(self, attn_weights_for_reg, base_reg_v, attn_mask=None, tgt_len=None):
        return do_av_mul_for_part(self, attn_weights_for_reg, base_reg_v, attn_mask, 'rr', tgt_len)

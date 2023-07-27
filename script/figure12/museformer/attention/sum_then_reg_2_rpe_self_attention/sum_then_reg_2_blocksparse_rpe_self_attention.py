import torch
import torch.nn.functional as F

from .sum_then_reg_2_rpe_self_attention import SumThenReg2RpeSelfAttention
from ...blocksparse import BlocksparseMatMul, BlocksparseSoftmax
from ...kernels.range_fill import range_fill


class SumThenRegBlocksparse2RpeSelfAttention(SumThenReg2RpeSelfAttention):
    def __init__(self, *args, block_size=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size

        num_heads_arange = torch.arange(self.num_heads)[:, None, None, None]
        self.register_buffer('num_heads_arange', num_heads_arange, persistent=False)
        block_size_arange = torch.arange(self.block_size)[None, None, :, None]
        self.register_buffer('block_size_arange', block_size_arange, persistent=False)

    def do_qk_scores_for_sum(self, base_sum_q, base_reg_k, bsz, sum_len, reg_len, attn_mask=None):
        # base_sum_q: (sum_len, bsz, heads, head_dim)
        # base_reg_k: (reg_len, bsz, heads, head_dim)
        tgt_len = sum_len
        src_len = reg_len
        if tgt_len <= 0 or src_len <= 0:  # basically this is for the situation where no summary is used
            return {'sr': base_sum_q.new_empty(0, self.block_size, self.block_size)}

        attn_mask = attn_mask['sr']
        attn_scores = []
        for idx in range(bsz):
            sample_layout = attn_mask[idx][0]  # (1, tgt_block, src_block)
            sample_attn_scores = self.do_sample_qk_scores_base(
                sample_layout, base_sum_q[:, idx], base_reg_k[:, idx],
                tgt_len, src_len, 'sum', idx
            )
            attn_scores.append(sample_attn_scores)
        attn_scores = {'sr': attn_scores}
        return attn_scores

    def do_sample_qk_scores_base(
        self, sample_layout, sample_tgt, sample_src,
        tgt_len, src_len, tgt_label, sample_idx
    ):
        # sample_layout: (1, tgt_block, src_block)

        assert sample_tgt.shape == (tgt_len, self.num_heads, self.head_dim)
        assert sample_src.shape == (src_len, self.num_heads, self.head_dim)
        assert sample_layout.shape == (1, tgt_len // self.block_size, src_len // self.block_size), \
            str(sample_layout.shape) + ' %d %d' % (tgt_len // self.block_size, src_len // self.block_size)
        sdd_matmul_key = (tgt_label, 'sdd_matmul', self.layer_sv, sample_idx)
        if sdd_matmul_key in self.instant_pocket:
            sdd_matmul = self.instant_pocket[sdd_matmul_key]
        else:
            sdd_matmul = BlocksparseMatMul(sample_layout, self.block_size, 'sdd',
                                           device=sample_tgt.device, trans_b=True)
            self.instant_pocket[sdd_matmul_key] = sdd_matmul

        sample_attn_scores = sdd_matmul(
            sample_tgt.transpose(0, 1)[:, None],  # (heads, 1, sum_len, head_dim)
            sample_src.transpose(0, 1)[:, None],  # (heads, 1, reg_len, head_dim)
        )  # (heads, head_selected_blocks, block, block)
        assert sample_attn_scores.shape[1] == int(sample_layout[0].sum())

        return sample_attn_scores

    def select_and_do_r_proj(self, rel_indices):
        """

        :param rel_indices: relation list of part dict
        :return:
        """

        r_embed_list = []
        r_modified_indices = []
        for rel_idx, one_rel_indices in enumerate(rel_indices):
            # Collect those indices for one kind of relative positions that are used for all the samples and real_parts
            one_rel_used_indices = []
            for real_part in one_rel_indices:  # ('sr', 'rx')
                one_rel_part_indices = one_rel_indices[real_part]
                if one_rel_part_indices is None:
                    continue
                for sample_rel_indices in one_rel_part_indices:
                    assert sample_rel_indices.shape[1:] == (self.block_size, self.block_size)
                    sample_used_rel_indices = torch.unique(sample_rel_indices)  # (num_unique,)
                    one_rel_used_indices.append(sample_used_rel_indices)
            one_rel_used_indices = torch.cat(one_rel_used_indices).unique()

            rel_selected_embed = F.embedding(one_rel_used_indices, self.rel_embeddings[rel_idx], padding_idx=0)

            rel_proj = getattr(self, 'rel%d_proj' % rel_idx, None)
            if rel_proj is not None:
                rel_selected_embed = rel_proj(rel_selected_embed)

            label_transform = range_fill(
                torch.stack((one_rel_used_indices, one_rel_used_indices + 1), dim=-1),
                torch.arange(len(one_rel_used_indices), device=one_rel_used_indices.device),
                self.num_rel_embeddings[rel_idx], 0
            )

            one_r_indices = {}
            for real_part in one_rel_indices:
                one_rel_part_indices = one_rel_indices[real_part]
                if one_rel_part_indices is None:
                    one_r_indices[real_part] = None
                    continue
                samples_r_indices = []
                for sample_rel_indices in one_rel_part_indices:
                    sample_r_indices = label_transform[sample_rel_indices]
                    samples_r_indices.append(sample_r_indices)
                one_r_indices[real_part] = samples_r_indices

            r_embed_list.append(rel_selected_embed)
            r_modified_indices.append(one_r_indices)

        return r_embed_list, r_modified_indices

    def add_rpe_for_sum(self, attn_scores_for_sum, rel_sum_qs, r_list, rel_indices, bsz, sum_len, attn_mask=None):
        attn_mask = attn_mask['sr']  # samples list of tuple (layout, block_mask)
        attn_scores_for_sum = attn_scores_for_sum['sr']  # samples list of (heads, head_selected_blocks, block, block)
        attn_scores_for_sum_with_rpe = [item for item in attn_scores_for_sum]
        for rel_idx in range(self.num_relation_types):
            r_indices = rel_indices[rel_idx]
            r_indices = r_indices['sr']

            if r_indices is None:
                continue

            r_embed = r_list[rel_idx].view(-1, self.num_heads, self.head_dim)  \
                # (num_selected_pos, heads, head_dim)
            r_sum_qs = rel_sum_qs[rel_idx]  # (sum_len, bsz, heads, head_dim)
            temp_r = torch.einsum("ibhd,jhd->bhij", r_sum_qs, r_embed)  # (bsz, heads, sum_len, num_selected_distance)
            for sample_idx in range(bsz):
                sample_r = temp_r[sample_idx]  # (heads, sum_len, num_selected_distance)
                sample_r_indices = r_indices[sample_idx]  # (head_selected_blocks, block, block)

                sample_layout = attn_mask[sample_idx][0]

                temp_rpe = self.indexing_sample_rpe_base(
                    sample_r, sample_r_indices, sample_layout, 'sum', sum_len, sample_idx
                )

                attn_scores_for_sum_with_rpe[sample_idx] = attn_scores_for_sum_with_rpe[sample_idx] + temp_rpe

        attn_scores_for_sum_with_rpe = {
            'sr': attn_scores_for_sum_with_rpe
        }

        return attn_scores_for_sum_with_rpe

    def indexing_sample_rpe_base(self, sample_r, sample_r_indices, sample_layout, tgt_label, tgt_len, sample_idx):
        # sample_r: (heads, tgt_len, num_selected_distance)
        # sample_r_indices: (head_selected_blocks, block, block)
        # sample_layout: (1, tgt_block, src_block)
        sample_r = sample_r.view(self.num_heads, tgt_len // self.block_size, self.block_size, -1)
        sample_tgt_block_ids = sample_layout[0].nonzero()[:, 0]  # (head_selected_blocks,)

        temp_rpe = sample_r[
            self.num_heads_arange,  # (8, 1, 1, 1)
            sample_tgt_block_ids[None, :, None, None],  # (1, 4, 1, 1)
            self.block_size_arange,  # (1, 1, block_size, 1)
            sample_r_indices[None],  # (1, head_selected_blocks, block, block)
        ]  # (head, head_selected_blocks, block, block)

        return temp_rpe

    def do_masking_for_sum(self, attn_scores_for_sum, attn_mask):
        return attn_scores_for_sum

    def do_attn_softmax_for_sum(self, attn_scores_for_sum, attn_mask=None):
        return self.do_attn_softmax_base(attn_scores_for_sum, attn_mask, 'sr')

    def do_attn_softmax_base(self, attn_scores_for_real_part, attn_mask, real_part):
        attn_scores_for_real_part = attn_scores_for_real_part[real_part]  \
            # samples list of (heads, head_selected_blocks, block, block)
        attn_mask = attn_mask[real_part]  # samples list of layout, block_mask
        bsz = len(attn_mask)
        result = [None] * bsz
        for sample_idx in range(bsz):
            sample_attn_mask = attn_mask[sample_idx]
            if sample_attn_mask is None:
                continue
            sample_layout, sample_block_mask = sample_attn_mask
            if sample_block_mask.dtype == torch.uint8:
                sample_block_mask = sample_block_mask.eq(1)
            assert sample_block_mask.dtype == torch.bool
            result[sample_idx] = attn_scores_for_real_part[sample_idx].masked_fill(sample_block_mask[None], -10000)

            softmax_label = (real_part, 'softmax', self.layer_sv, sample_idx)
            if softmax_label in self.instant_pocket:
                softmax = self.instant_pocket[softmax_label]
            else:
                softmax = BlocksparseSoftmax(sample_layout, self.block_size)
                self.instant_pocket[softmax_label] = softmax

            result[sample_idx] = softmax(result[sample_idx])

        result = {real_part: result}
        return result

    def do_sample_av_mul_base(self, sample_attn_weights, sample_v, sample_layout, real_part, sample_idx):
        sample_v = sample_v.transpose(0, 1)[:, None]  # (head, 1, reg_len, head_dim)
        dsd_matmul_key = (real_part, 'dsd_matmul', self.layer_sv, sample_idx)
        if dsd_matmul_key in self.instant_pocket:
            dsd_matmul = self.instant_pocket[dsd_matmul_key]
        else:
            dsd_matmul = BlocksparseMatMul(sample_layout, self.block_size, 'dsd',
                                           device=sample_v.device)
            self.instant_pocket[dsd_matmul_key] = dsd_matmul

        sample_out = dsd_matmul(sample_attn_weights, sample_v)  # (head, 1, tgt_len, head_dim)
        sample_out = sample_out.permute(2, 1, 0, 3)  # (tgt_len, 1, head, head_dim)

        return sample_out

    def do_av_mul_for_sum(self, attn_weights_for_sum, base_reg_v, attn_mask=None):
        attn_weights_for_sum = attn_weights_for_sum['sr']  # samples list of (head, head_selected_blocks, block, block)
        bsz = len(attn_weights_for_sum)
        attn_mask = attn_mask['sr']
        result = []
        for sample_idx in range(bsz):
            sample_reg_v = base_reg_v[:, sample_idx]
            sample_attn_weights = attn_weights_for_sum[sample_idx]  # (head, head_selected_blocks, block, block)
            sample_layout = attn_mask[sample_idx][0]

            sample_out = self.do_sample_av_mul_base(sample_attn_weights, sample_reg_v, sample_layout, 'sr', sample_idx)

            result.append(sample_out)

        return result

    def do_qk_scores_for_reg(
        self,
        base_reg_q, base_sum_k2, base_reg_k,
        bsz, sum_len, reg_len,
        attn_mask=None
    ):
        # base_reg_q: (reg_len, bsz, num_heads, head_dim)
        # base_sum_k2: (sum_len, bsz, num_heads, head_dim)
        # base_reg_k: (reg_len, bsz, num_heads, head_dim)
        tgt_len = reg_len
        src_len = sum_len + reg_len
        assert tgt_len > 0 and src_len > 0

        tgt = base_reg_q
        if sum_len == 0:
            src = base_reg_k
        else:
            src = torch.cat((base_sum_k2, base_reg_k), dim=0)  # (sum_len + reg_len, bsz, num_heads, head_dim)

        attn_mask = attn_mask['rx']
        attn_scores = []
        for idx in range(bsz):
            sample_layout = attn_mask[idx][0]  # (1, tgt_block, src_block)
            sample_attn_scores = self.do_sample_qk_scores_base(
                sample_layout, tgt[:, idx], src[:, idx],
                tgt_len, src_len, 'reg', idx
            )
            attn_scores.append(sample_attn_scores)
        attn_scores = {'rx': attn_scores}
        return attn_scores

    def add_rpe_for_reg(self, attn_scores_for_reg, rel_reg_qs, r_list, rel_indices,
                        bsz, reg_len, attn_mask=None):
        attn_mask = attn_mask['rx']  # samples list of tuple (layout, block_mask)
        attn_scores_for_reg = attn_scores_for_reg['rx']  # samples list of (heads, head_selected_blocks, block, block)
        attn_scores_for_reg_with_rpe = [item for item in attn_scores_for_reg]
        for rel_idx in range(self.num_relation_types):
            r_indices = rel_indices[rel_idx]
            r_indices = r_indices['rx']

            if r_indices is None:
                continue

            r_embed = r_list[rel_idx].view(-1, self.num_heads, self.head_dim) \
                # (num_selected_pos, heads, head_dim)
            r_reg_qs = rel_reg_qs[rel_idx]  # (reg_len, bsz, heads, head_dim)
            temp_r = torch.einsum("ibhd,jhd->bhij", r_reg_qs, r_embed)  # (bsz, heads, reg_len, num_selected_distance)
            for sample_idx in range(bsz):
                sample_r = temp_r[sample_idx]  # (heads, reg_len, num_selected_distance)
                sample_r_indices = r_indices[sample_idx]  # (head_selected_blocks, block, block)

                sample_layout = attn_mask[sample_idx][0]

                temp_rpe = self.indexing_sample_rpe_base(
                    sample_r, sample_r_indices, sample_layout, 'reg', reg_len, sample_idx
                )

                attn_scores_for_reg_with_rpe[sample_idx] = attn_scores_for_reg_with_rpe[sample_idx] + temp_rpe

        attn_scores_for_reg_with_rpe = {
            'rx': attn_scores_for_reg_with_rpe
        }

        return attn_scores_for_reg_with_rpe

    def do_masking_for_reg(self, attn_scores_for_reg, attn_mask):
        return attn_scores_for_reg

    def do_attn_softmax_for_reg(self, attn_scores_for_reg, attn_mask=None):
        return self.do_attn_softmax_base(attn_scores_for_reg, attn_mask, 'rx')

    def do_av_mul_for_reg(self, attn_weights_for_reg, base_sum_v2, base_reg_v, attn_mask=None):
        attn_weights_for_reg = attn_weights_for_reg['rx']  # samples list of (head, head_selected_blocks, block, block)
        bsz = len(attn_weights_for_reg)
        attn_mask = attn_mask['rx']
        result = []
        if base_sum_v2 is None:
            value = base_reg_v
        else:
            value = torch.cat((base_sum_v2, base_reg_v), dim=0)
        for sample_idx in range(bsz):
            sample_v = value[:, sample_idx]
            sample_attn_weights = attn_weights_for_reg[sample_idx]  # (head, head_selected_blocks, block, block)
            sample_layout = attn_mask[sample_idx][0]

            sample_out = self.do_sample_av_mul_base(sample_attn_weights, sample_v, sample_layout, 'rx', sample_idx)

            result.append(sample_out)

        return result

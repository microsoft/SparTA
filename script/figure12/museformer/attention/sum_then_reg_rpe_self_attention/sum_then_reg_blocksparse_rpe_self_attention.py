import torch

from .sum_then_reg_rpe_self_attention import SumThenRegRpeSelfAttention
from ...blocksparse import BlocksparseMatMul, BlocksparseSoftmax
from ...data_structures.four_dim_pocket import FourDimPocket


class SumThenRegBlocksparseRpeSelfAttention(SumThenRegRpeSelfAttention):
    def __init__(self, *args, block_size=16, same_for_all_heads=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.same_for_all_heads = same_for_all_heads

        num_heads_arange = torch.arange(self.num_heads)
        self.register_buffer('num_heads_arange', num_heads_arange, persistent=False)

        self.pocket = FourDimPocket()

    def do_qk_scores(
        self,
        base_q, base_k,
        bsz, query_sum_len, query_reg_len, key_sum_len, key_reg_len,
        attn_mask=None,
        query_label=None,
        **kwargs
    ):
        query_len = query_sum_len + query_reg_len
        key_len = key_sum_len + key_reg_len
        if query_len <= 0 or key_len <= 0:
            return base_q.new_empty(0, self.block_size, self.block_size)

        layout, _ = attn_mask

        base_q = base_q.view(query_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)[None]  \
            # (1, bsz * num_heads, all_seq_len, head_dim)
        base_k = base_k.view(key_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)[None] \
            # (1, bsz * num_heads, all_seq_len, head_dim)
        sdd_matmul_key = (query_label, 'sdd_matmul', self.layer_idx)
        if sdd_matmul_key in self.pocket:
            sdd_matmul = self.pocket[sdd_matmul_key]
        else:
            raise NotImplementedError("Not prepared for instant initialization.")
            sdd_matmul = BlocksparseMatMul(layout, self.block_size, device=base_q.device, mode='sdd', trans_b=True)

        try:
            attn_weights = sdd_matmul(base_q, base_k)  # (1, num_squares, block_size, block_size)
        except RuntimeError:
            print(base_q.shape, base_k.shape, layout.shape)
            raise
        assert layout.sum() == attn_weights.shape[1], (int(layout.sum()), attn_weights.shape[1])
        attn_weights = attn_weights.squeeze(0)  # (num_squares, block_size, block_size)

        return attn_weights

    def add_rpe(self, attn_mask, attn_weights, rel_qs, r_list, rel_indices, valid_parts,
                bsz, sum_len, reg_len):
        if attn_weights.shape[0] <= 0:
            return attn_weights

        blocksparse_token_rpe = None
        layout = attn_mask[0]
        # temp_rpe: Optional[torch.Tensor] = None
        for idx in range(self.num_relation_types):
            rel_q = rel_qs[idx]  # (all_seq_len, bsz, num_heads, head_dim)
            r = r_list[idx]  # (num_used_rel, num_heads, head_dim)

            for part_name, query_slice, key_slice in (
                ('ss', slice(None, sum_len), slice(None, sum_len)),
                ('sr', slice(None, sum_len), slice(sum_len, None)),
                ('rs', slice(sum_len, None), slice(None, sum_len)),
                ('rr', slice(sum_len, None), slice(sum_len, None)),
            ):
                if part_name not in valid_parts:
                    continue
                if self.rel_settings[part_name] is True or self.rel_settings[part_name][idx]:
                    if part_name.find('s') != -1 and sum_len <= 0:
                        continue

                    temp_q = rel_q[query_slice]  # (part_query_len, bsz, num_heads, head_dim)
                    rel_index = rel_indices[idx]  \
                        # (bsz, all_seq_len, all_seq_len) or dict of ((bsz, len, len) or str or tuple)
                    rel_index = rel_index[part_name]  # (bsz, part_query_len, part_key_len)
                    if isinstance(rel_index, torch.Tensor):
                        rel_index = rel_index.permute(1, 2, 0)  # (part_query_len, part_key_len, bsz)

                    if rel_index == 'blocksparse':
                        raise NotImplementedError('Must compute rel_index for blocksparse ahead')
                        # assert part_name == 'rr'
                        # num_reg_chunks = reg_len // self.block_size
                        # rel_index = generate_blocksparse_token_rpe_indices(
                        #     layout[:, -num_reg_chunks:, -num_reg_chunks:], self.block_size, bsz, reg_len, self.num_heads,
                        #     self.rel_max_positions[idx],
                        #     same_for_all_heads=self.same_for_all_heads
                        # )  # tuple (rel_ids, row_indices, blocks_rel_pos_ids)

                    if isinstance(rel_index, tuple):
                        rel_ids, row_indices, blocks_rel_pos_ids = rel_index
                        assert bsz == 1
                        assert self.same_for_all_heads
                        assert part_name == 'rr'
                        # assume bsz == 1
                        # rel_ids: (num_used_rels,)
                        # row_indices: (sample_blocks, num_heads, block_size)
                        # blocks_rel_pos_ids: (sample_blocks, num_heads, block_size, block_size)
                        temp_r = torch.einsum("ibnd,jnd->ijbn", temp_q, r)  \
                            # (part_query_len, num_used_rel, bsz, num_heads)
                        del temp_q
                        # print(temp_r.shape)
                        # print(row_indices.shape)
                        # print(blocks_rel_pos_ids.shape)
                        blocksparse_token_rpe = temp_r[
                            row_indices[:, :, None, None],
                            blocks_rel_pos_ids[:, :, :, None],
                            0,  # since bsz == 1
                            self.num_heads_arange[None, None, None]
                        ].permute(3, 0, 1, 2)
                    elif isinstance(rel_index, torch.Tensor):
                        raise NotImplementedError
                        temp_r = torch.einsum("ibnd,jnd->ijbn", temp_q, r)  # (part_query_len, num_rel, bsz, num_heads)
                        del temp_q
                        temp_r = embedding_indexing(temp_r, rel_index)  # (part_query_len, part_key_len, bsz, num_heads)
                        temp_r = temp_r.view(*(temp_r.shape[:2]), -1).permute(2, 0, 1)
                        if temp_rpe is None:
                            temp_rpe = attn_weights.new_zeros(bsz * self.num_heads, all_seq_len, all_seq_len)
                        temp_rpe[:, query_slice, key_slice] = temp_rpe[:, query_slice, key_slice] + temp_r
                    else:
                        raise ValueError(type(rel_index))
                    del temp_r

        # if temp_rpe is not None:
        #     num_all_squares = all_seq_len // self.block_size
        #     temp_rpe = temp_rpe.view(
        #         bsz * self.num_heads,
        #         num_all_squares, self.block_size,
        #         num_all_squares, self.block_size,
        #     ).permute(
        #         0, 1, 3, 2, 4
        #     ).masked_select(
        #         layout[:, :, :, None, None]
        #     ).view(-1, self.block_size, self.block_size)
        #     attn_weights = attn_weights + temp_rpe

        if blocksparse_token_rpe is not None:
            assert valid_parts == ('rs', 'rr') or valid_parts == ('rr', 'rs')
            num_sum_chunks = sum_len // self.block_size
            num_reg_chunks = reg_len // self.block_size
            rr_layout_selection = layout.clone()
            assert rr_layout_selection.shape == (self.num_heads, num_reg_chunks, num_sum_chunks + num_reg_chunks)
            rr_layout_selection[:, :, :num_sum_chunks] = False
            rr_layout_selection = rr_layout_selection[layout][:, None, None].expand(
                -1, self.block_size, self.block_size
            )  # (num_squares, block_size, block_size)
            attn_weights[rr_layout_selection] = attn_weights[rr_layout_selection] + blocksparse_token_rpe.reshape(-1)

        return attn_weights

    def do_masking(
        self, attn_weights, attn_mask,
        bsz, query_len, key_len
    ):
        return attn_weights

    def do_attn_softmax_float(self, attn_weights, attn_mask=None, **kwargs):
        attn_weights = attn_weights.float()  # (num_selected_squares, block, block)
        if attn_weights.shape[0] <= 0:
            return attn_weights
        if attn_mask is not None:
            _, block_mask = attn_mask
            attn_weights = attn_weights.masked_fill(block_mask, -1e9)
        layout, _ = attn_mask
        softmax_key = ('softmax', self.layer_idx)
        if softmax_key in self.pocket:
            softmax = self.pocket[softmax_key]
        else:
            softmax = BlocksparseSoftmax(layout, self.block_size)
        attn_weights_float = softmax(attn_weights[None])  # (1, num_selected_squares, block, block)
        return attn_weights_float

    def do_av_mul(
        self, attn_probs, v,
        bsz, tgt_len, src_len,
        attn_mask=None,
        query_label=None
    ):
        if tgt_len <= 0 or src_len <= 0:
            return attn_probs.new_empty(0, bsz, self.embed_dim)
        layout, _ = attn_mask
        dsd_matmul_key = (query_label, 'dsd_matmul', self.layer_idx)
        if dsd_matmul_key in self.pocket:
            dsd_matmul = self.pocket[dsd_matmul_key]
        else:
            dsd_matmul = BlocksparseMatMul(layout, self.block_size, device=attn_probs.device, mode='dsd')
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(1, 0)[None]
        attn = dsd_matmul(attn_probs, v)[0]  # (bsz * num_heads, tgt_len, head_dim)
        assert attn.shape == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn = attn.view(bsz, self.num_heads, tgt_len, self.head_dim).permute(2, 0, 1, 3)  \
            # (all_seq_len, bsz, num_heads, head_dim)
        attn = attn.reshape(tgt_len, bsz, self.embed_dim)
        return attn

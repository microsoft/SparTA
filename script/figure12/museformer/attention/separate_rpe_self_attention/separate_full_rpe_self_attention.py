import torch
from fairseq import utils
from .separate_rpe_self_attention import SeparateRpeSelfAttention, embedding_indexing


def skew(x, dis_embed_near_to_far=False):
    # x: (tgt_len, num_rels, bsz, num_heads)
    raise NotImplementedError("Need to check.")
    tgt_len, num_rels, bsz, num_heads = x.shape
    assert tgt_len == num_rels - 1
    if dis_embed_near_to_far:
        new_x = torch.empty_like(x)
        new_x[:, 0] = x[:, 0]
        new_x[:, 1:] = x[:, 1:].flip((1,))
        x = new_x
        del new_x
    bsz_heads = bsz * num_heads
    x = x.view(tgt_len, num_rels, bsz_heads)
    x = torch.as_strided(
        x, (num_rels, tgt_len, bsz_heads),
        (tgt_len * bsz_heads, bsz_heads, 1)
    )
    x = x[1:]
    return x.view(tgt_len, tgt_len, bsz, num_heads)


class SeparateFullRpeSelfAttention(SeparateRpeSelfAttention):
    def do_qk_scores(
        self,
        base_q, base_k, rel_qs, r_list, rel_indices,
        bsz, sum_len, reg_len, all_seq_len,
        *args, **kwargs
    ):
        attn_weights = torch.einsum("ibnd,jbnd->ijbn", base_q, base_k)  # (all_seq_len, all_seq_len, bsz, num_heads)
        for idx in range(self.num_relation_types):
            rel_q = rel_qs[idx]  # (all_seq_len, bsz, num_heads, head_dim)
            r = r_list[idx]  # (num_rel, num_heads, head_dim)

            for part_name, query_slice, key_slice in (
                ('ss', slice(None, sum_len), slice(None, sum_len)),
                ('sr', slice(None, sum_len), slice(sum_len, None)),
                ('rs', slice(sum_len, None), slice(None, sum_len)),
                ('rr', slice(sum_len, None), slice(sum_len, None)),
            ):
                if part_name.find('s') != -1 and sum_len <= 0:
                    continue
                if self.rel_settings[part_name] is True or self.rel_settings[part_name][idx]:
                    temp_q = rel_q[query_slice]  # (part_query_len, bsz, num_heads, head_dim)
                    rel_index = rel_indices[idx]  # (bsz, all_seq_len, all_seq_len) or dict of (bsz, len, len)
                    if isinstance(rel_index, dict):
                        rel_index = rel_index[part_name]  # (bsz, part_query_len, part_key_len)
                        if isinstance(rel_index, str):  # skewing
                            pass
                        else:
                            rel_index = rel_index.permute(1, 2, 0)  # (part_query_len, part_key_len, bsz)
                    else:
                        rel_index = rel_index.permute(1, 2, 0)  # (all_seq_len, all_seq_len, bsz)
                        rel_index = rel_index[query_slice, key_slice]  # (part_query_len, part_key_len, bsz)

                    if isinstance(rel_index, torch.Tensor):
                        temp_max_pos = rel_index.max()
                        temp_r = r[:temp_max_pos + 1]  # (num_rel, num_heads, head_dim)
                        # num_rel = temp_r.shape[0]
                        temp_r = torch.einsum("ibnd,jnd->ijbn", temp_q, temp_r)  \
                            # (part_query_len, num_rel, bsz, num_heads)
                        del temp_q
                        temp_r = embedding_indexing(temp_r, rel_index)  # (part_query_len, part_key_len, bsz, num_heads)
                    else:
                        part_query_len = temp_q.shape[0]
                        temp_r = r[:part_query_len + 1]  # (part_query_len + 1, num_heads, head_dim)
                        temp_r = torch.einsum("ibnd,jnd->ijbn", temp_q, temp_r)  \
                            # (part_query_len, part_query_len + 1, bsz, num_heads)
                        temp_r = skew(temp_r, dis_embed_near_to_far=(rel_index == 'skew_flip'))
                    attn_weights[query_slice, key_slice] = attn_weights[query_slice, key_slice] + temp_r
                    del temp_r

        attn_weights = attn_weights.permute(2, 3, 0, 1)  # (bsz, num_heads, all_seq_len, all_seq_len)
        assert attn_weights.shape == (bsz, self.num_heads, all_seq_len, all_seq_len)

        return attn_weights

    def do_masking(
        self, attn_weights, attn_mask, bsz, all_seq_len
    ):
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, all_seq_len, all_seq_len)
            elif attn_mask.ndim == 3:
                first_dim = attn_mask.shape[0]
                if first_dim == self.num_heads:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    assert first_dim == bsz * self.num_heads
                    attn_mask = attn_mask.view(bsz, self.num_heads, all_seq_len, all_seq_len)
                del first_dim
            else:
                raise ValueError
            if attn_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
            else:
                raise NotImplementedError(attn_mask.dtype)

        return attn_weights

    def do_attn_softmax_float(self, attn_weights, *args, **kwargs):
        attn_weights_float = utils.softmax(
            attn_weights, dim=-1
        )
        return attn_weights_float

    def do_av_mul(
        self, attn_probs, v,
        bsz, all_seq_len,
        **kwargs
    ):
        attn = torch.einsum('bhts,sbhd->sbhd', attn_probs, v)
        assert attn.shape == (all_seq_len, bsz, self.num_heads, self.head_dim)
        attn = attn.view(all_seq_len, bsz, self.embed_dim)
        return attn

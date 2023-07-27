import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules.fairseq_dropout import FairseqDropout

from museformer.embedding.embedding_weight_generation import \
    generate_sinusoid_position_embedding_with_padding, generate_randomly_initialized_position_embedding_with_padding
from museformer.tools import computation_tools


def select_used_rel_embed(rel_embed, rel_ids):
    # rel_embed: (num_rel_pos + 1, heads, head_dim)
    # rel_ids: (num_used_rels,) relative position ids that needs computing
    num_rels = rel_embed.shape[0]
    assert rel_ids.ndim == 1
    num_used_rels = rel_ids.shape[0]
    if num_used_rels < num_rels:
        rel_embed = F.embedding(rel_ids, rel_embed, padding_idx=0)
    elif num_used_rels > num_rels:
        raise ValueError
    return rel_embed


def embedding_indexing(x, indices):
    """
    Indexing position embeddings.
    :param x: (tgt_len, num_embed, bsz, num_heads)
    :param indices: (tgt_len, src_len, bsz)
    :return: (tgt_len, src_len, bsz, num_heads)
    """
    device = x.device
    tgt_len, num_embed, bsz, _ = x.shape

    tgt_len_arange = torch.arange(tgt_len, device=device)
    bsz_arange = torch.arange(bsz, device=device)

    r = x[
        tgt_len_arange[:, None, None],
        indices,
        bsz_arange[None, None]
    ]

    return r


class SumThenRegRpeSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,

        rel_max_positions,  # list of int, indicating the max distance for multiple relative position embeddings

        layer_idx=None,
        dropout=0.0,  # attention dropout

        query_proj_bias=True,
        key_proj_bias=True,
        value_proj_bias=True,
        sum_key2_proj_bias=True,
        sum_value2_proj_bias=True,
        out_proj_bias=True,
        add_different_kqv_bias_for_sum_and_reg=False,

        share_query_proj=False,
        share_key_proj=False,
        share_value_proj=False,
        share_out_proj=False,

        key_rel_proj_bias=True,
        add_global_rel_bias=True,
        rel_dims=None,  # Support designate different dimensions for relative position embeddings
        use_sinusoid_relative_embeddings=True,
        rel_embeddings=None,
        rel_settings=None,

        same_for_all_heads=False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.same_for_all_heads = same_for_all_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "attention_embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.num_relation_types = len(rel_max_positions)
        self.rel_max_positions = rel_max_positions

        if rel_settings is None:
            # rel_settings = {
            #     'rr': False,
            #     'rs': False,
            #     'sr': False,
            #     'ss': False,
            # }
            raise ValueError("Must assign rel_settings.")
        else:
            for key in ('rr', 'rs', 'sr', 'ss'):
                rel_setting = rel_settings[key]
                assert isinstance(rel_setting, bool) or len(rel_setting) == self.num_relation_types
        self.rel_settings = rel_settings

        if add_different_kqv_bias_for_sum_and_reg:
            query_proj_bias = False
            key_proj_bias = False
            value_proj_bias = False
            for proj_name in ('query', 'key', 'value'):
                for target in ('sum', 'reg'):
                    bias_tensor = torch.zeros(self.embed_dim)
                    self.register_parameter('%s_%s_bias' % (target, proj_name),
                                            nn.Parameter(bias_tensor, requires_grad=True))

        if rel_dims is None:
            rel_dims = (None,) * self.num_relation_types
        else:
            assert len(rel_dims) == self.num_relation_types
        self.rel_dims = [self.embed_dim if item is None else item for item in rel_dims]

        self.same_rel_dim = None
        if self.num_relation_types == 0 or \
            not all([self.rel_dims[idx] == self.rel_dims[0]] for idx in range(1, self.num_relation_types)):
            pass
        else:
            self.same_rel_dim = self.rel_dims[0]

        self.reg_query_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=query_proj_bias)
        if share_query_proj:
            self.sum_query_proj = self.reg_query_proj
        else:
            self.sum_query_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=query_proj_bias)
        self.reg_key_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=key_proj_bias)
        if share_key_proj:
            self.sum_key_proj = self.reg_key_proj
        else:
            self.sum_key_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=key_proj_bias)
        self.reg_value_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=value_proj_bias)
        if share_value_proj:
            self.sum_value_proj = self.reg_value_proj
        else:
            self.sum_value_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=value_proj_bias)

        self.sum_key2_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=sum_key2_proj_bias)
        self.sum_value2_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=sum_value2_proj_bias)

        for idx in range(self.num_relation_types):
            if (
                'no_rel_pos_proj' in self.rel_settings['others'][idx]
                and self.rel_settings['others'][idx]['no_rel_pos_proj']
            ):
                key_rel_proj = None
            else:
                rel_dim = self.rel_dims[idx]
                key_rel_proj = nn.Linear(rel_dim, embed_dim, bias=key_rel_proj_bias)
            setattr(self, 'key_rel%d_proj' % idx, key_rel_proj)

        if self.num_relation_types > 0 and add_global_rel_bias:
            self.global_rel_base_sum_bias = nn.Parameter(torch.empty(self.num_heads, self.head_dim), requires_grad=True)
            self.global_rel_base_reg_bias = nn.Parameter(torch.empty(self.num_heads, self.head_dim), requires_grad=True)
            for idx in range(self.num_relation_types):
                enabled_for_sum = False
                for key in ('sr', 'ss'):
                    rel_setting = self.rel_settings[key]
                    if rel_setting is True or rel_setting[idx]:
                        enabled_for_sum = True
                        break
                if enabled_for_sum:
                    temp_bias = nn.Parameter(torch.empty(self.num_heads, self.head_dim), requires_grad=True)
                    setattr(self, 'global_rel%d_sum_bias' % idx, temp_bias)
                enabled_for_reg = False
                for key in ('rr', 'rs'):
                    rel_setting = self.rel_settings[key]
                    if rel_setting is True or rel_setting[idx]:
                        enabled_for_reg = True
                        break
                if enabled_for_reg:
                    temp_bias = nn.Parameter(torch.empty(self.num_heads, self.head_dim), requires_grad=True)
                    setattr(self, 'global_rel%d_reg_bias' % idx, temp_bias)
            raise NotImplementedError('Temporarily disable add_aglobal_rel_bias for sanity check.')

        self.reg_out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=out_proj_bias)
        if share_out_proj:
            self.sum_out_proj = self.reg_out_proj
        else:
            self.sum_out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=out_proj_bias)

        if rel_embeddings is None:
            rel_embeddings = (None,) * self.num_relation_types
        else:
            assert len(rel_embeddings) == self.num_relation_types

        if self.num_relation_types > 0:
            all_max_position = max(self.rel_max_positions)
            all_sinusoid_embedding_weight = generate_sinusoid_position_embedding_with_padding(
                all_max_position, self.rel_dims[0]
            )
            if isinstance(use_sinusoid_relative_embeddings, bool):
                use_sinusoid_relative_embeddings = (use_sinusoid_relative_embeddings,) * self.num_relation_types
            for idx, embedding_weight in enumerate(rel_embeddings):
                max_position = self.rel_max_positions[idx]
                rel_dim = self.rel_dims[idx]
                rel_sinusoid = use_sinusoid_relative_embeddings[idx]
                if embedding_weight is not None:
                    assert embedding_weight.shape == (max_position + 1, rel_dim)
                    setattr(self, 'rel%d_embedding' % idx, embedding_weight)
                else:
                    if rel_sinusoid:
                        if rel_dim == self.rel_dims[0]:
                            embedding_weight = all_sinusoid_embedding_weight
                            self.register_buffer('rel%d_embedding' % idx, embedding_weight[:max_position + 1],
                                                 persistent=False)
                        else:
                            embedding_weight = generate_sinusoid_position_embedding_with_padding(
                                max_position, rel_dim
                            )
                            self.register_buffer('rel%d_embedding' % idx, embedding_weight, persistent=False)
                    else:
                        embedding_weight = generate_randomly_initialized_position_embedding_with_padding(
                            max_position, rel_dim
                        )
                        self.register_parameter('rel%d_embedding' % idx,
                                                nn.Parameter(embedding_weight, requires_grad=True))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.reg_query_proj.weight, gain=1 / math.sqrt(2))
        if id(self.sum_query_proj) != id(self.reg_query_proj):
            nn.init.xavier_uniform_(self.sum_query_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.reg_key_proj.weight, gain=1 / math.sqrt(2))
        if id(self.sum_key_proj) != id(self.reg_key_proj):
            nn.init.xavier_uniform_(self.sum_key_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.reg_value_proj.weight, gain=1 / math.sqrt(2))
        if id(self.sum_value_proj) != id(self.reg_value_proj):
            nn.init.xavier_uniform_(self.sum_value_proj.weight, gain=1 / math.sqrt(2))

        nn.init.xavier_uniform_(self.sum_key2_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.sum_value2_proj.weight, gain=1 / math.sqrt(2))

        for idx in range(self.num_relation_types):
            key_rel_proj = getattr(self, "key_rel%d_proj" % idx, None)
            if key_rel_proj is None:
                continue
            key_rel_proj_weight = key_rel_proj.weight
            nn.init.xavier_uniform_(key_rel_proj_weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.reg_out_proj.weight)
        if id(self.sum_out_proj) != id(self.reg_out_proj):
            nn.init.xavier_uniform_(self.sum_out_proj.weight)

        global_rel_bias_names = ('global_rel_base_sum_bias', 'global_rel_base_reg_bias') + tuple(
            ['global_rel%d_%s_bias' % (idx, ts) for idx in range(self.num_relation_types) for ts in ('sum', 'reg')]
        )
        for bias_name in global_rel_bias_names:
            bias = getattr(self, bias_name, None)
            if bias is not None:
                nn.init.zeros_(bias)

        if self.reg_out_proj.bias is not None:
            nn.init.constant_(self.reg_out_proj.bias, 0.0)
        if id(self.sum_out_proj) != id(self.reg_out_proj) and self.sum_out_proj.bias is not None:
            nn.init.constant_(self.sum_out_proj.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,  # (all_seq_len, bsz, embed_dim)  all_seq_len == sum_len + reg_len
        sum_len,
        reg_len,
        rel_indices,  # relation list of dict of parts (bsz, len, len)
        key_padding_mask=None,  # (bsz, all_seq_len)
        attn_mask=None,
        need_weights: bool = False,
        need_head_weights: bool = False,
        *args, **kwargs,
    ):
        if need_head_weights:
            need_weights = True

        all_seq_len, bsz, embed_dim = x.size()
        assert sum_len + reg_len == all_seq_len
        assert embed_dim == self.embed_dim

        assert len(rel_indices) == self.num_relation_types
        for rel_index in rel_indices:
            assert isinstance(rel_index, dict)

        if key_padding_mask is not None:
            # assert key_padding_mask.shape == (bsz, all_seq_len)
            raise NotImplementedError("Please combine key_padding_mask into attn_mask ahead.")
        del key_padding_mask

        base_q, rel_qs = self.do_q_proj(
            x,
            bsz, sum_len, reg_len, all_seq_len
        )  # q: (all_seq_len, bsz, num_heads, head_dim)  rel_qs: (all_seq_len, bsz, num_heads, head_dim)

        base_k = self.do_k_proj(
            x,
            bsz, sum_len, reg_len, all_seq_len
        )  # k: (all_seq_len, bsz, num_heads, head_dim)

        r_list = self.do_r_proj(rel_indices)  # r: (num_rel, num_heads, head_dim)

        # --- attention weights for sum as query ---
        sum_base_q = base_q[:sum_len]

        sum_attn_weights = self.do_qk_scores(
            sum_base_q, base_k,
            bsz, sum_len, 0, sum_len, reg_len,
            attn_mask=attn_mask[0],
            query_label='sum'
        )
        sum_attn_weights = self.add_rpe(attn_mask[0], sum_attn_weights, rel_qs, r_list, rel_indices, ('ss', 'sr'),
                                        bsz, sum_len, reg_len)
        sum_attn_weights.mul_(self.scaling)

        sum_attn_weights = self.do_masking(
            sum_attn_weights, attn_mask[0],
            bsz, sum_len, all_seq_len
        )

        sum_attn_weights_float = self.do_attn_softmax_float(sum_attn_weights, attn_mask=attn_mask[0], query_label='sum')
        sum_attn_weights = sum_attn_weights_float.type_as(sum_attn_weights)
        sum_attn_probs = self.dropout_module(sum_attn_weights)
        sum_attn_weights = None

        v = self.do_v_proj(
            x,
            bsz, sum_len, reg_len, all_seq_len
        )  # (all_seq_len, bsz, num_heads, head_dim)

        sum_attn = self.do_av_mul(
            sum_attn_probs, v,
            bsz, sum_len, all_seq_len,
            attn_mask=attn_mask[0],
            query_label='sum'
        )  # (sum_len, bsz, embed_dim)

        sum_key = self.sum_key2_proj(sum_attn).view(sum_len, bsz, self.num_heads, self.head_dim)
        sum_value = self.sum_value2_proj(sum_attn).view(sum_len, bsz, self.num_heads, self.head_dim)

        # --- attention weights for reg as query ---
        reg_base_q = base_q[sum_len:]

        base_k = torch.cat(
            (sum_key, base_k[sum_len:]), dim=0
        )
        reg_attn_weights = self.do_qk_scores(
            reg_base_q, base_k,
            bsz, 0, reg_len, sum_len, reg_len,
            attn_mask=attn_mask[1],
            query_label='reg'
        )
        reg_attn_weights = self.add_rpe(attn_mask[1], reg_attn_weights, rel_qs, r_list, rel_indices, ('rs', 'rr'),
                                        bsz, sum_len, reg_len)
        reg_attn_weights.mul_(self.scaling)

        reg_attn_weights = self.do_masking(
            reg_attn_weights, attn_mask[1],
            bsz, sum_len, all_seq_len
        )

        reg_attn_weights_float = self.do_attn_softmax_float(reg_attn_weights, attn_mask=attn_mask[1], query_label='reg')
        reg_attn_weights = reg_attn_weights_float.type_as(reg_attn_weights)
        reg_attn_probs = self.dropout_module(reg_attn_weights)
        reg_attn_weights = None

        v = torch.cat(
            (sum_value, v[sum_len:]), dim=0
        )

        reg_attn = self.do_av_mul(
            reg_attn_probs, v,
            bsz, reg_len, all_seq_len,
            attn_mask=attn_mask[1],
            query_label='reg',
        )  # (sum_len, bsz, embed_dim)

        attn = self.do_out_proj((sum_attn, reg_attn), sum_len, reg_len)

        # attn_weights = attn_weights_float.float()
        # attn_weights_save = attn_weights.detach().cpu()
        # with open('attn_weights_at_layer_%d.pt' % self.layer_idx, 'wb') as f:
        #     torch.save(attn_weights_save, f)
        # del attn_weights_save

        if need_weights:
            raise NotImplementedError
            attn_weights = self.do_attn_weights(
                attn_weights_float,
                bsz, all_seq_len,
                need_head_weights=need_head_weights
            )
        else:
            attn_weights = None
        # if not need_weights:
        #     attn_weights = None

        return attn, attn_weights
        # (all_seq_len, bsz, embed_dim)
        # (bsz, num_heads, all_seq_len, all_seq_len) or (bsz, all_seq_len, all_seq_len)

    def do_q_proj(
        self,
        x,
        bsz, sum_len, reg_len, all_seq_len
    ):
        q = computation_tools.bi_projection(self.sum_query_proj, self.reg_query_proj, x, sum_len, reg_len) \
            # (all_seq_len, bsz, embed_dim)
        sum_q_bias = getattr(self, 'sum_query_bias', None)
        reg_q_bias = getattr(self, 'reg_query_bias', None)
        q = computation_tools.may_bi_add(q, sum_q_bias, reg_q_bias, sum_len, reg_len)

        q = q.view(all_seq_len, bsz, self.num_heads, self.head_dim)  # (all_seq_len, bsz, num_heads, head_dim)

        base_q = computation_tools.may_bi_add(
            q,
            getattr(self, 'global_rel_base_sum_bias', None),
            getattr(self, 'global_rel_base_reg_bias', None),
            sum_len, reg_len
        )  # (all_seq_len, bsz, embed_dim)
        rel_qs = []
        for idx in range(self.num_relation_types):
            rel_q = computation_tools.may_bi_add(
                q,
                getattr(self, 'global_rel%d_sum_bias' % idx, None),
                getattr(self, 'global_rel%d_reg_bias' % idx, None),
                sum_len, reg_len
            )
            rel_qs.append(rel_q)

        return base_q, rel_qs

    def do_k_proj(
        self,
        x,
        bsz, sum_len, reg_len, all_seq_len,
    ):
        base_k = computation_tools.bi_projection(self.sum_key_proj, self.reg_key_proj, x, sum_len, reg_len)
        sum_k_bias = getattr(self, 'sum_key_bias', None)
        reg_k_bias = getattr(self, 'reg_key_bias', None)
        base_k = computation_tools.may_bi_add(base_k, sum_k_bias, reg_k_bias, sum_len, reg_len)
        base_k = base_k.view(all_seq_len, bsz, self.num_heads, self.head_dim)

        return base_k

    def do_r_proj(
        self, rel_indices
    ):
        r_list = []
        for idx in range(self.num_relation_types):
            rel_index = rel_indices[idx]  # (bsz, all_seq_len, all_seq_len) or dict of (bsz, part_len_1, part_len_2)
            rel_embedding = getattr(self, 'rel%d_embedding' % idx)  # (num_rel, dim)
            # collect which distances should be computed
            position_selection = 0
            assert isinstance(rel_index, dict), str(type(rel_index))
            for temp_key in rel_index:  # ss sr rs rr
                key_rel_index = rel_index[temp_key]
                if key_rel_index is None or self.rel_settings[temp_key][idx] is False:
                    # None means no need to compute for this part (ss, sr, rs, rr)
                    continue
                if isinstance(key_rel_index, torch.Tensor):
                    position_selection = max(key_rel_index.max().item(), position_selection)
                elif isinstance(key_rel_index, tuple):
                    # (position_selection, row_indices, blocks_indices) for blocksparse
                    assert list(rel_index.keys()) == ['rr']
                    position_selection, _, blocks_rel_pos_ids = key_rel_index
                    if position_selection is not None:
                        break
                    else:
                        position_selection = blocks_rel_pos_ids.max().item()
                        break
                # Todo: skew support
                else:
                    raise ValueError(type(key_rel_index))
            del temp_key, key_rel_index

            # Todo: consider skewing
            if isinstance(position_selection, int):
                rel_embedding = select_used_rel_embed(
                    rel_embedding,
                    torch.arange(position_selection + 1, device=rel_embedding.device)
                )
            elif isinstance(position_selection, torch.Tensor):
                rel_embedding = select_used_rel_embed(rel_embedding, position_selection)
            else:
                raise ValueError(type(position_selection))

            r_proj = getattr(self, 'key_rel%d_proj' % idx)
            if r_proj is not None:
                r = r_proj(rel_embedding)
            else:
                r = rel_embedding
            r = r.view(-1, self.num_heads, self.head_dim)  # (num_rel, num_heads, head_dim)
            r_list.append(r)

        return r_list

    def do_qk_scores(
        self,
        base_q, base_k,
        bsz, query_sum_len, query_reg_len, key_sum_len, key_reg_len,
        **kwargs
    ):
        raise NotImplementedError

    def add_rpe(self, attn_mask, attn_weights, rel_qs, r_list, rel_indices, valid_parts,
                bsz, sum_len, reg_len):
        raise NotImplementedError

    def do_masking(
        self, attn_weights, attn_mask,
        bsz, query_len, key_len
    ):
        raise NotImplementedError

    def do_v_proj(
        self, x, bsz, sum_len, reg_len, all_seq_len
    ):
        v = computation_tools.bi_projection(self.sum_value_proj, self.reg_value_proj, x, sum_len, reg_len) \
            # (all_seq_len, bsz, embed_dim)
        sum_v_bias = getattr(self, 'sum_value_bias', None)
        reg_v_bias = getattr(self, 'reg_value_bias', None)
        v = computation_tools.may_bi_add(v, sum_v_bias, reg_v_bias, sum_len, reg_len)
        v = v.view(all_seq_len, bsz, self.num_heads, self.head_dim)  # (all_seq_len, bsz, num_heads, head_dim)
        return v

    def do_attn_softmax_float(self, attn_weights, *args, **kwargs):
        raise NotImplementedError

    def do_av_mul(
        self, attn_probs, v,
        bsz, tgt_len, src_len,
        **kwargs
    ):
        raise NotImplementedError

    def do_out_proj(
        self,
        attn, sum_len, reg_len
    ):
        attn = computation_tools.bi_projection(
            self.sum_out_proj, self.reg_out_proj,
            attn, sum_len, reg_len
        )
        return attn

    def do_attn_weights(self, attn_weights,
                        bsz, all_seq_len,
                        need_head_weights=False):
        attn_weights = attn_weights.float()
        attn_weights = attn_weights.view(
            bsz, self.num_heads, all_seq_len, all_seq_len
        )
        if not need_head_weights:
            # average attention weights over heads
            attn_weights = attn_weights.mean(dim=1)
        return attn_weights

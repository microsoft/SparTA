from ..tools import arg_tools


def add_args(parser):
    parser.add_argument('--attn-query-proj-bias', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-key-proj-bias', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-value-proj-bias', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-out-proj-bias', type=arg_tools.str_bool_with_default_error)

    # valid: sum_then_reg, v2.1
    parser.add_argument('--attn-sum-key2-proj-bias', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-sum-value2-proj-bias', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-share-key2-value2-proj-weight', type=arg_tools.str_bool_with_default_error)

    parser.add_argument('--add-different-kqv-bias-for-sum-and-reg', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--add-different-out-bias-for-sum-and-reg', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-share-query-proj', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-share-key-proj', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-share-value-proj', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-share-out-proj', type=arg_tools.str_bool_with_default_error)

    # valid: sum_then_reg_3
    parser.add_argument('--attn-share-reg-kv-proj', type=arg_tools.str_bool_with_default_error)

    # parser.add_argument('--attn-key-rel-proj-bias', type=arg_tools.str_bool_with_default_error)
    # parser.add_argument('--attn-add-global-rel-bias', type=arg_tools.str_bool_with_default_error)


def create_separate_attention(
    *args, implementation='mask', block_size=16, layer_idx=None, same_for_all_heads=False, **kwargs
):
    if implementation == 'mask':
        from .separate_rpe_self_attention.separate_full_rpe_self_attention import SeparateFullRpeSelfAttention
        return SeparateFullRpeSelfAttention(*args, layer_idx=layer_idx, **kwargs)
    elif implementation == 'blocksparse':
        from .separate_rpe_self_attention.separate_blocksparse_rpe_self_attention import \
            SeparateBlocksparseRpeSelfAttention
        return SeparateBlocksparseRpeSelfAttention(*args, block_size=block_size,
                                                   layer_idx=layer_idx,
                                                   same_for_all_heads=same_for_all_heads,
                                                   **kwargs)
    else:
        raise NotImplementedError(implementation)


def create_sum_then_reg_attention(
    *args, implementation='mask', block_size=16, layer_idx=None, same_for_all_heads=False, **kwargs
):
    if implementation == 'mask':
        raise NotImplementedError
    elif implementation == 'blocksparse':
        from .sum_then_reg_rpe_self_attention.sum_then_reg_blocksparse_rpe_self_attention import \
            SumThenRegBlocksparseRpeSelfAttention
        return SumThenRegBlocksparseRpeSelfAttention(*args, block_size=block_size,
                                                     layer_idx=layer_idx,
                                                     same_for_all_heads=same_for_all_heads,
                                                     **kwargs)
    else:
        raise NotImplementedError(implementation)


def create_attention_v2_s1(
    *args, implementation='mask', block_size=16, **kwargs
):
    if implementation == 'blocksparse':
        from .self_attention_v2s1.blocksparse_rpe_self_attention_v2s1 import BlocksparseRpeSelfAttentionV2S1
        return BlocksparseRpeSelfAttentionV2S1(
            *args, block_size=block_size, **kwargs
        )
    elif implementation == 'sparta':
        from .self_attention_v2s1.sparta_rpe_self_attention_v2s1 import SpartaRpeSelfAttentionV2S1
        return SpartaRpeSelfAttentionV2S1(
            *args, **kwargs
        )
    elif implementation == 'mask':
        from .self_attention_v2s1.full_rpe_self_attention_v2s1 import SpartaRpeSelfAttentionV2S1
        return SpartaRpeSelfAttentionV2S1(
            *args, **kwargs
        )
    elif implementation == 'triton':
        from .self_attention_v2s1.triton_rpe_self_attention_v2s1 import TritonRpeSelfAttentionV2S1
        return TritonRpeSelfAttentionV2S1(
            *args, **kwargs
        )
    elif implementation == 'deepspeed':
        from .self_attention_v2s1.deepspeed_rpe_self_attention_v2s1 import DeepSpeedRpeSelfAttentionV2S1
        return DeepSpeedRpeSelfAttentionV2S1(
            *args, block_size=block_size, **kwargs
        )
    return NotImplementedError(implementation)


def create_sum_then_reg_2_attention(
    *args, implementation='blocksparse', block_size=16, **kwargs
):
    if implementation == 'blocksparse':
        from .sum_then_reg_2_rpe_self_attention.sum_then_reg_2_blocksparse_rpe_self_attention import \
            SumThenRegBlocksparse2RpeSelfAttention
        return SumThenRegBlocksparse2RpeSelfAttention(
            *args, block_size=block_size, **kwargs
        )
    return NotImplementedError(implementation)


def create_sum_then_reg_3_attention(
    *args, implementation='blocksparse', block_size=16, **kwargs
):
    if implementation == 'blocksparse':
        from .sum_then_reg_3_rpe_self_attention.sum_then_reg_3_blocksparse_rpe_self_attention import \
            SumThenRegBlocksparse3RpeSelfAttention
        return SumThenRegBlocksparse3RpeSelfAttention(
            *args, block_size=block_size, **kwargs
        )
    return NotImplementedError


def create_attention(
    *args, attention_mode='simu', **kwargs
):
    if attention_mode == 'simu':  # v1
        raise NotImplementedError
        return create_separate_attention(*args, **kwargs)
    elif attention_mode == 'sum_then_reg':  # v2
        raise NotImplementedError
        return create_sum_then_reg_attention(*args, **kwargs)
    elif attention_mode == 'v2s1':  # v2.1
        return create_attention_v2_s1(*args, **kwargs)
    elif attention_mode == 'sum_then_reg_2':  # v3
        raise NotImplementedError
        return create_sum_then_reg_2_attention(*args, **kwargs)
    elif attention_mode == 'sum_then_reg_3':  # v5
        return create_sum_then_reg_3_attention(*args, **kwargs)
    else:
        raise NotImplementedError(attention_mode)

import torch


def generate_blocksparse_token_rpe_indices(layout, block_size,
                                           bsz, seq_len, heads, num_rel_pos,
                                           same_for_all_heads=False):
    if bsz > 1:
        raise NotImplementedError
    if not same_for_all_heads:
        raise NotImplementedError('Not Implemented for different head settings. Memory concerned.')
    device = layout.device
    num_chunks = seq_len // block_size
    assert num_chunks * block_size == seq_len
    assert layout.shape == (bsz * heads, num_chunks, num_chunks)
    origin_layout = layout.view(bsz, heads, num_chunks, num_chunks)[:, 0: 1].reshape(
        bsz, num_chunks, num_chunks
    )
    origin_layout.tril_()
    # layout = origin_layout.sum(dim=0).gt(0)
    # nonzero = layout.nonzero(as_tuple=False)
    # offset = nonzero[:, 0] - nonzero[:, 1]
    # min_dis = (-block_size + 1) + offset * block_size
    # min_dis[min_dis.lt(0)] = 0
    # max_dis = (block_size - 1) + offset * block_size
    # max_dis[max_dis >= num_rel_pos] = num_rel_pos - 1
    # dis_ranges = torch.stack((min_dis, max_dis + 1), dim=-1)
    # dis_ranges.add_(1)  # (num_flat_blocks, 2)
    # rel_pos_labeling = range_fill(dis_ranges,
    #                               torch.ones(dis_ranges.shape[0], dtype=torch.long, device=device),
    #                               num_rel_pos + 1, 0).bool()
    # rel_pos_labeling[0] = True
    # rel_ids = rel_pos_labeling.nonzero(as_tuple=False).squeeze(1)
    # rel_pos_labeling_transform = rel_pos_labeling.cumsum(dim=0) - 1
    # rel_pos_labeling_transform[~rel_pos_labeling] = 0  # (need check)

    layout_nonzero = origin_layout.nonzero(as_tuple=False)
    row_indices = torch.arange(0, block_size, device=device)[None] + layout_nonzero[:, 1][:, None] * block_size  \
        # (num_selected_blocks_one_head, block_size)
    # row_indices = row_indices[:, None].expand(-1, heads, block_size)  \
    #     # (num_selected_blocks_one_head, heads, block_size)
    block_offsets = layout_nonzero[:, 1] - layout_nonzero[:, 2]
    block_size_arange = torch.arange(0, block_size, device=device)
    zero_rel_pos_ids = block_size_arange[:, None] - block_size_arange[None]
    del block_size_arange
    blocks_rel_pos_ids = block_offsets[:, None, None] * block_size + zero_rel_pos_ids[None]
    del zero_rel_pos_ids
    blocks_rel_pos_ids[blocks_rel_pos_ids.lt(0)] = -1
    blocks_rel_pos_ids[blocks_rel_pos_ids.ge(num_rel_pos)] = num_rel_pos - 1
    blocks_rel_pos_ids.add_(1)
    # blocks_rel_pos_ids = rel_pos_labeling_transform[blocks_rel_pos_ids]  \
    #     # (num_selected_blocks_one_head, block_size, block_size)
    # blocks_rel_pos_ids = blocks_rel_pos_ids[:, None].expand(-1, heads, block_size, block_size)  \
    #     # (num_selected_blocks_one_head, heads, block_size, block_size)
    return None, row_indices, blocks_rel_pos_ids
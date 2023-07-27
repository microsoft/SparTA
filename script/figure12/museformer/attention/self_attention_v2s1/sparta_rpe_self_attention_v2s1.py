import torch

from ..common.sparta.dynamic_sparta import DynamicSparseAttention2 as DynamicSparseAttention

from .rpe_self_attention_v2s1 import RpeSelfAttentionV2S1

from ..common.blocksparse_common_operations.rel_proj.rel_proj_1 import select_and_do_r_proj

import numpy as np
from nvitop import Device


def get_gpu_info():
    devices = Device.all()  # or `Device.cuda.all()` to use CUDA ordinal instead
    for device in devices:
        processes = device.processes()  # type: Dict[int, GpuProcess]
        sorted_pids = sorted(processes.keys())

        print(device)
        print(f"  - Fan speed:       {device.fan_speed()}%")
        print(f"  - Temperature:     {device.temperature()}C")
        print(f"  - GPU utilization: {device.gpu_utilization()}%")
        print(f"  - Total memory:    {device.memory_total_human()}")
        print(f"  - Used memory:     {device.memory_used_human()}")
        print(f"  - Free memory:     {device.memory_free_human()}")
        print(f"  - Processes ({len(processes)}): {sorted_pids}")
        for pid in sorted_pids:
            print(f"    - {processes[pid]}")
        print("-" * 120)

class SpartaRpeSelfAttentionV2S1(RpeSelfAttentionV2S1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.num_relation_types == 0, 'sparta does not support rpe'

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
        del sum_token_ids
        assert len(rel_indices) == self.num_relation_types
        assert bsz == 1

        # ===== Obtain attn_mask =====
        attn_mask_label = 'attn_mask_sv%d' % self.layer_sv
        attn_mask = self.instant_pocket[attn_mask_label]
        sx_attn_mask = attn_mask['sx']  # (bsz, 1, sum_len, sum_len + reg_len)
        if sx_attn_mask is not None:
            assert sx_attn_mask.shape == (bsz, 1, sum_len, sum_len + reg_len)
            sx_attn_mask = sx_attn_mask.view(sum_len, sum_len + reg_len)  # (sum_len, sum_len + reg_len)
        rx_attn_mask = attn_mask['rx']  # (bsz, 1, reg_len, sum_len + reg_len)
        if rx_attn_mask is not None:
            assert rx_attn_mask.shape == (bsz, 1, reg_len, sum_len + reg_len)
            rx_attn_mask = rx_attn_mask.view(reg_len, sum_len + reg_len)  # (reg_len, sum_len + reg_len)
        self.instant_pocket[attn_mask_label] = {'sx': None, 'rx': None}  # delete masks

        # ===== Rel Indices =====
        # r_list, rel_indices = self.select_and_do_r_proj(rel_indices)
        # r_list: relation list of dict of real_parts (num_selected_rel, embed_dim)
        # rel_indices_for_sum: relation list of dict of real_parts

        # ===== Summarize =====
        base_reg_k = self.reg_key_proj(reg_x)
        bias = getattr(self, 'reg_key_bias', None)
        if bias is not None:
            base_reg_k = base_reg_k + bias
        base_reg_k = base_reg_k.view(reg_len, bsz, self.num_heads, self.head_dim)
        # base_reg_k: (reg_len, bsz, num_heads, head_dim)

        base_reg_v = self.reg_value_proj(reg_x)
        bias = getattr(self, 'reg_value_bias', None)
        if bias is not None:
            base_reg_v = base_reg_v + bias
        base_reg_v = base_reg_v.view(reg_len, bsz, self.num_heads, self.head_dim)
        # base_reg_v: (reg_len, bsz, num_heads, head_dim)

        if sum_len > 0:
            base_sum_q = self.sum_query_proj(sum_x)
            bias = getattr(self, 'sum_query_bias', None)
            if bias is not None:
                base_sum_q = base_sum_q + bias
            base_sum_q = base_sum_q.view(sum_len, bsz, self.num_heads, self.head_dim)
            # rel_sum_qs = [base_sum_q for _ in range(self.num_relation_types)]
            # base_sum_q: (sum_len, bsz, num_heads, head_dim)  rel_sum_qs: list of (sum_len, bsz, num_heads, head_dim)

            base_sum_k = self.sum_key_proj(sum_x)
            bias = getattr(self, 'sum_key_bias', None)
            if bias is not None:
                base_sum_k = base_sum_k + bias
            base_sum_k = base_sum_k.view(sum_len, bsz, self.num_heads, self.head_dim)

            base_sum_v = self.sum_value_proj(sum_x)
            bias = getattr(self, 'sum_value_bias', None)
            if bias is not None:
                base_sum_v = base_sum_v + bias
            base_sum_v = base_sum_v.view(sum_len, bsz, self.num_heads, self.head_dim)

            sx_label = (attn_mask_label, 'sparta', 'sx')
            if sx_label in self.instant_pocket:
                sx_spa = self.instant_pocket[sx_label]
            else:
                assert sx_attn_mask is not None
                sx_spa = DynamicSparseAttention((~sx_attn_mask).int())
                self.instant_pocket[sx_label] = sx_spa

            temp_q = base_sum_q  # (sum_len, bsz, heads, head_dim)
            temp_k = torch.cat((base_sum_k, base_reg_k), dim=0)  # (sum_len + reg_len, bsz, heads, head_dim)
            temp_k.mul_(self.scaling)
            temp_v = torch.cat((base_sum_v, base_reg_v), dim=0)  # (sum_len + reg_len, bsz, heads, head_dim)
            assert not temp_q.isnan().any()
            assert not temp_k.isnan().any()
            assert not temp_v.isnan().any()
            sum_x2 = sx_spa(
                temp_q.permute(1, 2, 0, 3).float(),
                temp_k.permute(1, 2, 0, 3).float(),
                temp_v.permute(1, 2, 0, 3).float()
            ).to(temp_q.dtype)  # (bsz, heads, sum_len, head_dim)
            # assert sum_x2.shape == (bsz, self.num_heads, sum_len, self.head_dim)
            try:
                assert not sum_x2.isnan().any()
            except AssertionError:
                sparse_mask = sx_spa.sparse_mask
                sparse_mask_allzero = sparse_mask.sum(dim=1).eq(0)
                # print(sparse_mask.sum(dim=1).eq(0))
                # print(sparse_mask.sum(dim=1).eq(0).any())
                nan_mask = sum_x2.squeeze(1).sum(dim=1).isnan()
                print(sparse_mask_allzero)
                print(nan_mask)
                print(sum_x2.squeeze(1).isnan().nonzero())
                print(sparse_mask.nonzero().tolist())
                # print(torch.equal(sparse_mask_allzero, nan_mask))
                print(sum_len, reg_len)
                save_dict = {
                    'q': temp_q.permute(1, 2, 0, 3),
                    'k': temp_k.permute(1, 2, 0, 3),
                    'v': temp_v.permute(1, 2, 0, 3),
                    'mask': sparse_mask,
                    'result': sum_x2
                }
                torch.save(save_dict, 'error_save2.bin')
                print('saved')
                raise

            sum_x2 = sum_x2.contiguous().permute(2, 0, 1, 3)
            assert sum_x2.shape == (sum_len, bsz, self.num_heads, self.head_dim)
            sum_x2 = sum_x2.reshape(sum_len, bsz, self.embed_dim)

            del sx_spa, sx_label, temp_q, temp_k, temp_v, sx_attn_mask

            if self.share_key2_value2_proj_weight:
                base_sum_k2 = self.sum_key2_proj(sum_x2)
                base_sum_v2 = base_sum_k2
                sum_key2_bias = getattr(self, 'sum_key2_bias', None)
                if sum_key2_bias is not None:
                    base_sum_k2 = base_sum_k2 + sum_key2_bias
                sum_value2_bias = getattr(self, 'sum_value2_bias', None)
                if sum_value2_bias is not None:
                    base_sum_v2 = base_sum_v2 + sum_value2_bias
                base_sum_k2 = base_sum_k2.view(sum_len, bsz, self.num_heads, self.head_dim)
                base_sum_v2 = base_sum_v2.view(sum_len, bsz, self.num_heads, self.head_dim)
            else:
                base_sum_k2 = self.sum_key2_proj(sum_x2).view(sum_len, bsz, self.num_heads, self.head_dim)
                base_sum_v2 = self.sum_value2_proj(sum_x2).view(sum_len, bsz, self.num_heads, self.head_dim)

        else:
            sum_x2 = reg_x.new_empty(0, bsz, self.embed_dim)
            base_sum_k2 = None
            base_sum_v2 = None

        # ===== Updating =====
        base_reg_q = self.reg_query_proj(reg_x)
        reg_query_bias = getattr(self, 'reg_query_bias', None)
        if reg_query_bias is not None:
            base_reg_q = base_reg_q + reg_query_bias
        base_reg_q = base_reg_q.view(reg_len, bsz, self.num_heads, self.head_dim)
        # rel_reg_qs = [base_reg_q for _ in range(self.num_relation_types)]

        rx_label = (attn_mask_label, 'sparta', 'rx')
        if rx_label in self.instant_pocket:
            rx_spa = self.instant_pocket[rx_label]
        else:
            assert rx_attn_mask is not None
            rx_spa = DynamicSparseAttention((~rx_attn_mask).int())
            self.instant_pocket[rx_label] = rx_spa

        temp_q = base_reg_q  # (reg_len, bsz, num_heads, head_dim)
        if base_sum_k2 is None:
            temp_k = base_sum_k2
        else:
            temp_k = torch.cat((base_sum_k2, base_reg_k), dim=0)  # (sum_len + reg_len, bsz, num_heads, head_dim)
        assert not base_sum_k2.isnan().any()
        assert not base_reg_k.isnan().any()
        temp_k.mul_(self.scaling)
        if base_sum_v2 is None:
            temp_v = base_reg_v
        else:
            temp_v = torch.cat((base_sum_v2, base_reg_v), dim=0)  # (sum_len + reg_len, bsz, num_heads, head_dim)
        assert not temp_q.isnan().any()
        # assert not temp_k.isnan().any()
        # assert not temp_v.isnan().any()
        reg_output = rx_spa(
            temp_q.permute(1, 2, 0, 3).float(),
            temp_k.permute(1, 2, 0, 3).float(),
            temp_v.permute(1, 2, 0, 3).float(),
        ).to(temp_q.dtype)  # (bsz, num_heads, reg_len, head_dim)
        assert not reg_output.isnan().any()

        reg_output = reg_output.contiguous().permute(2, 0, 1, 3)  # (reg_len, bsz, num_heads, head_dim)

        del temp_q, temp_k, temp_v, rx_spa, rx_label, rx_attn_mask

        # ----- gate to combine sum_output and reg_output -----
        reg_output = reg_output.reshape(reg_len, bsz, self.embed_dim)
        reg_output = self.reg_out_proj(reg_output)
        reg_out_bias = getattr(self, 'reg_out_bias', None)
        if reg_out_bias is not None:
            reg_output = reg_output + reg_out_bias
        if not self.no_sum_out and self.num_summary > 0:
            sum_output = self.sum_out_proj(sum_x2)
            sum_out_bias = getattr(self, 'sum_out_bias', None)
            if sum_out_bias is not None:
                sum_output = sum_output + sum_out_bias
        else:
            sum_output = None

        if need_weights:
            raise NotImplementedError
        else:
            attn_weights = None

        # if np.random.rand() > 0.99:
        #     get_gpu_info()

        return (sum_output, reg_output), attn_weights
        # (sum_len, bsz, embed_dim)  (reg_len, bsz, embed_dim)
        # None, (bsz, num_heads, all_seq_len, all_seq_len) or (bsz, all_seq_len, all_seq_len)

    # ============= Interfaces =============
    def select_and_do_r_proj(self, rel_indices):
        """

        :param rel_indices: relation list of real_part dict
        :return:
        """
        return select_and_do_r_proj(self, rel_indices)

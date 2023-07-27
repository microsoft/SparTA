import torch

from sparta.opset.sparse_opbase import SparseOPBase
from sparta.opset.dynamic_sparse_attention import DynamicSparseAttentionFunction
from sparta.opset.bcsr_converter import BcsrConverter


class DynamicSparseAttention2(SparseOPBase):
    """
    The Sparse Attention module that support the dynamic sparse pattern.
    """

    def __init__(self, sparse_mask):
        """
        Parameters
        ----------
        HEAD_NUM: int
            The number of heads of the sparse attention
        max_seq_length: int
            The maximum length of the input sequence
        global_mode: bool
            If use the global sparse pattern, if true, then all the sparse_attention
            instance share the same sparse pattern to get the better performance
        """
        super(DynamicSparseAttention2, self).__init__()
        assert isinstance(sparse_mask, torch.Tensor)
        sparse_mask = sparse_mask.int()
        self.sparse_mask = sparse_mask
        self.t_sparse_mask = sparse_mask.t().contiguous()
        # currently only support 32 x 64
        self.block_size_h = 32
        self.block_size_w = 32
        self.converter = BcsrConverter()
        self.inter_result = None  # tensor to store the internal results

        bcsr_row, bcsr_col, bcsr_row_pos, bcsr_val_mask, bcsr_block_index = self.converter(
            sparse_mask, sparse_mask.to(torch.float), self.block_size_h, self.block_size_w, True
        )

        grad_bcsr_row, grad_bcsr_col, grad_bcsr_row_pos, grad_bcsr_val_mask, grad_bcsr_block_index = self.converter(
            self.t_sparse_mask,
            self.t_sparse_mask.to(torch.float),
            self.block_size_w,
            self.block_size_h,
            True
        )

        n_row = sparse_mask.size(0) // self.block_size_h
        block_nnz = bcsr_row[n_row].item()
        self.bcsr_row = bcsr_row
        self.bcsr_col = bcsr_col
        self.bcsr_row_pos = bcsr_row_pos
        self.bcsr_val_mask = bcsr_val_mask
        self.bcsr_block_index = bcsr_block_index
        self.n_row = n_row
        self.block_nnz = block_nnz
        self.tgt_len, self.src_len = sparse_mask.shape
        self.grad_bcsr_row = grad_bcsr_row
        self.grad_bcsr_col = grad_bcsr_col
        self.grad_bcsr_row_pos = grad_bcsr_row_pos
        self.grad_bcsr_val_mask = grad_bcsr_val_mask
        self.grad_bcsr_block_index = grad_bcsr_block_index

    def forward(self, Q, K, V):
        """
        Q, K, V are the output tensors of the corresponding
        projection linear layers.
        (bsz, heads, eq_len, head)dim)
        """
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        csr_row, csr_col, csr_row_pos, csr_value_mask, csr_block_index = \
            self.bcsr_row, self.bcsr_col, self.bcsr_row_pos, self.bcsr_val_mask, self.bcsr_block_index
        grad_scr_row, grad_scr_col = self.grad_bcsr_row, self.grad_bcsr_col
        # n_row = self.n_row
        block_nnz = self.block_nnz
        # need create val each time
        # assert isinstance(Q, torch.Tensor)
        # assert isinstance(K, torch.Tensor)
        # assert isinstance(V, torch.Tensor)
        # Shape of tensor Q should be {Batchsize, sequence length, hidden dim}
        batch_size, head_num, tgt_len, hidden_dim = Q.shape
        src_len = K.size(2)
        err_msg = 'Currently, tgt_len and hidden_dim should be divisible by 32'
        assert tgt_len % 32 == 0, err_msg
        assert src_len % 32 == 0, err_msg
        assert hidden_dim % 32 == 0, err_msg
        assert tgt_len == self.tgt_len, "input tgt sequence length (%d) dose not match the given sparse pattern (%d)" % (
            tgt_len, self.tgt_len
        )
        assert src_len == self.src_len, "input src sequence length (%d) dose not match the given sparse pattern (%d)" % (
            src_len, self.src_len
        )
        assert K.shape == V.shape
        sparse_val_size = block_nnz * self.block_size_h * self.block_size_w
        if (
                self.inter_result is None or
                self.inter_result.numel() < batch_size * head_num * block_nnz * self.block_size_h * self.block_size_w
        ):
            self.inter_result = torch.zeros(
                batch_size * head_num * sparse_val_size, dtype=torch.float32, device=Q.device
            )
        result = DynamicSparseAttentionFunction.apply(
            Q, K, V,
            self.inter_result,
            csr_row,
            csr_col,
            csr_row_pos,
            csr_value_mask,
            csr_block_index,  # for backward
            grad_scr_row,  # for backward
            grad_scr_col,  # for backward
            block_nnz,
            head_num
        )

        return result

    def reference_forward(self, Q, K, V):
        """
        Calculate the reference result the sparse attention to test the correctness.
        """

        out_mask = self.sparse_mask
        add_mask = torch.zeros(out_mask.size()).to(Q.device)
        add_mask[out_mask == 0] = float('-inf')
        dots = torch.einsum('b h m k, b h n k -> b h m n', Q, K)
        added = torch.add(dots, add_mask)
        attn = added.softmax(dim=-1)
        nan_pos = torch.isnan(attn)
        attn = attn.masked_fill(nan_pos, 0.0)
        ref_out = torch.einsum('b h m n, b h n k -> b h m k', attn, V)

        return ref_out

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from SparTA.Common.Utils import convert_bcsr

class SparseLayout:
    def __init__(self):
        pass
    def _build_index(self):
        raise NotImplementedError

class LayoutTransformation:
    def __init__(self, ori_layout, new_layout):
        self.ori_layout = ori_layout
        self.new_layout = new_layout
    def _build_index(self):
        raise NotImplementedError

class BCSRSparseLayout(SparseLayout):
    def __init__(self, block_size, dense_mask):
        self.block_size = block_size
        self.dense_mask = dense_mask
        self._build_index()
    
    def _build_index(self):
        self.block_row_idx, self.block_col_idx, self.fine_graind_mask = convert_bcsr(self.dense_mask, self.dense_mask)

class BCSRLayoutTransformation(LayoutTransformation):
    def __init__(self, ori_layout, new_layout):
        super(BCSRLayoutTransformation, self).__init__(ori_layout, new_layout)
    
    def _build_index(self):
        # TODO support layout auto transformation in the future
        raise NotImplementedError
import os
from copy import deepcopy
import pickle
import logging

import numpy as np
import torch
from fairseq.data import data_utils, FairseqDataset
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset
from fairseq.data.indexed_dataset import MMapIndexedDatasetBuilder


logger = logging.getLogger(__name__)


def get_bar_chunk_points(seq: torch.Tensor, eob_index, begin_idx=0):
    # seq: (seq_len,)
    # eob_index: int
    is_complete_bar = seq[-1] == eob_index
    indices = seq.eq(eob_index).nonzero(as_tuple=False).squeeze(1)  # (num_bars,)
    indices = indices + 1
    indices = torch.cat(
        (indices.new_tensor([begin_idx]), indices), dim=0
    )
    len_seq = len(seq)
    if not is_complete_bar and len_seq > begin_idx:
        indices = torch.cat(
            (indices, indices.new_tensor([len_seq])), dim=0
        )
    return indices, is_complete_bar


class BarChunkSequenceDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset, src_dict, eob,
        eos_appended=True,
        offset=0,
    ):
        super().__init__(dataset)
        self.src_dict = src_dict
        self.eos_appended = eos_appended
        self.eob = eob
        self.offset = offset

    def __iter__(self):
        len_dataset = len(self)
        for idx in range(len_dataset):
            yield self[idx]

    def __getitem__(self, index):
        sample = self.dataset[index]  # all include eoc
        chunk_points, _ = get_bar_chunk_points(
            sample[:-1] if self.eos_appended else sample,
            self.eob, begin_idx=self.offset
        )
        return sample, chunk_points

    def collater(self, samples):
        raise NotImplementedError("Dataset class %s is not designed for collating samples." % self.__class__.__name__)


def ChunkSequenceDataset(
    dataset, src_dict,
    eob, eoc,
    chunking_scheme='bar_aware',
    eos_appended=True,
    dataset_name=None,
    cache_data_label=None,
    cache_sequence=None,
    offset=0
):
    if chunking_scheme == 'bar_aware':
        return BarChunkSequenceDataset(
            dataset, src_dict, eob,
            eos_appended=eos_appended,
            offset=offset
        )

    raise NotImplementedError(chunking_scheme)

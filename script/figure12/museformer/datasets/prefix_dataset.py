import numpy as np
import torch
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset


class BosPrefixDataset(BaseWrapperDataset):
    def __init__(self, dataset, bos_index):
        super().__init__(dataset)
        self.dataset = dataset
        self.bos_index = bos_index
        self.__default_prefix = torch.tensor([self.bos_index], dtype=torch.long)
        self._sizes = np.array(dataset.sizes) + 1

    def __getitem__(self, idx):
        item, chunk_points = self.dataset[idx]
        item = torch.cat([self.__default_prefix, item])
        return item, chunk_points + 1, 1

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        return self.dataset.num_tokens(index) + 1

    def size(self, index):
        return self._sizes[index]

    def collater(self, samples):
        raise NotImplementedError("Dataset class %s is not designed for collating samples." % self.__class__.__name__)


def PrefixDataset(
    dataset,
    bos_index=4,
    prefix_generation_scheme_name='default',
    dataset_name=None,
    cache_data_label=None
):
    if prefix_generation_scheme_name == 'default':
        return BosPrefixDataset(dataset, bos_index)

    raise NotImplementedError(prefix_generation_scheme_name)

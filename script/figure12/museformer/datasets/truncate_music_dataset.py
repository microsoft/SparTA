import torch
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset


class TruncateMusicDataset(BaseWrapperDataset):
    def __init__(self, dataset, truncated_length):
        super().__init__(dataset)
        assert truncated_length > 1
        self.truncated_length = truncated_length

        self._sizes = self.dataset.sizes.copy()
        self._sizes[self._sizes > self.truncated_length] = self.truncated_length

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if sample['src_length'] <= self.truncated_length:
            sample['num_complete_chunks'] = sample['num_chunks']
            return sample

        assert sample['num_pref'] < self.truncated_length
        sample['src_tokens'] = sample['src_tokens'][:self.truncated_length]
        sample['src_length'] = self.truncated_length
        chunk_points = sample['chunk_points']
        chunk_points = chunk_points[chunk_points.le(self.truncated_length)]
        if chunk_points[-1] == self.truncated_length:
            num_chunks = len(chunk_points) - 1
            num_complete_chunks = num_chunks
        else:
            num_chunks = len(chunk_points)
            num_complete_chunks = num_chunks - 1
            chunk_points = torch.cat((chunk_points, chunk_points.new_tensor([self.truncated_length])))
        sample['chunk_points'] = chunk_points
        sample['target'] = sample['target'][:self.truncated_length]
        sample['num_chunks'] = num_chunks
        sample['num_complete_chunks'] = num_complete_chunks
        return sample

    @property
    def sizes(self):
        return self._sizes

    def size(self, index):
        return self._sizes[index]

    def num_tokens(self, index):
        return self._sizes[index]

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        num_complete_chunks = [s['num_complete_chunks'] for s in samples]
        num_complete_chunks = torch.tensor(num_complete_chunks, dtype=torch.long)
        batch = super().collater(samples)
        batch['net_input']['num_complete_chunks'] = num_complete_chunks
        return batch

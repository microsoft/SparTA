import torch
from fairseq.data import data_utils
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset


class MusicMonolingualDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        pad_index=1,
    ):
        super().__init__(dataset)
        self._size = self.dataset.sizes - 1
        self.pad_index = pad_index

    @property
    def sizes(self):
        return self._size

    def __iter__(self):
        len_dataset = len(self)
        for idx in range(len_dataset):
            yield self[idx]

    def __getitem__(self, index):
        sample = self.dataset[index]
        src_tokens, chunk_points, num_pref = sample

        source = src_tokens[:-1]  # (pref + chunks)
        target = src_tokens[1:]  # (other_pref + chunks + eos)
        seq_len = source.shape[0]

        sample = {
            'src_tokens': source,
            'src_length': seq_len,
            'chunk_points': chunk_points,
            'num_pref': num_pref,
            'target': target,
            'num_chunks': len(chunk_points) - 1,
        }

        return sample

    def size(self, index):
        return self._size[index]

    def num_tokens(self, index):
        return self.size(index)

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        bsz = len(samples)
        src_tokens = [s['src_tokens'] for s in samples]
        src_lengths = [s['src_length'] for s in samples]
        chunk_points = [s['chunk_points'] for s in samples]
        num_pref = [s['num_pref'] for s in samples]
        target = [s['target'] for s in samples]
        num_chunks = [s['num_chunks'] for s in samples]
        ntokens = sum(src_lengths)

        src_tokens = data_utils.collate_tokens(
            src_tokens, self.pad_index,
        )
        src_lengths = torch.tensor(src_lengths, dtype=torch.long)
        chunk_points = data_utils.collate_tokens(
            chunk_points, 0
        )
        num_pref = torch.tensor(num_pref, dtype=torch.long)
        target = data_utils.collate_tokens(target, self.pad_index)
        num_chunks = torch.tensor(num_chunks, dtype=torch.long)

        batch = {
            'nsentences': bsz,
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'chunk_points': chunk_points,
                'num_pref': num_pref,
                'num_chunks': num_chunks,
            },
            'target': target,
        }

        return batch

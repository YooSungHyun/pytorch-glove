import torch
from torch.nn.utils.rnn import pad_sequence


class CustomDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(CustomDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        # make pad or something work for each step's batch
        center = list()
        k = list()
        cooccur_counts = list()

        for i in range(len(batch)):
            center.append(torch.tensor(batch[i]["i"]))
            k.append(torch.tensor(batch[i]["k"]))
            cooccur_counts.append(torch.tensor(batch[i]["cooccur_counts"]))

        _returns = {"center": torch.stack(center), "k": torch.stack(k), "cooccur_counts": torch.stack(cooccur_counts)}
        # _returns = {"input_ids", "attention_mask", "input_type_ids", "labels"}

        return _returns

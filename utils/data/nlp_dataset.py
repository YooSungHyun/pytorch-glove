import math
import numpy as np
from utils.abc_dataset import CustomDataset
import numpy
from collections.abc import Callable
from collections import defaultdict, Counter
import copy


class CBOWDataset(CustomDataset):
    def __init__(self, tot_comat: numpy.ndarray, vocab: dict = None, transform: Callable = None):
        self.tot_comat = tot_comat
        self.vocab = vocab
        non_zero_indices = np.argwhere(self.tot_comat != 0)
        self.raw_data = list()
        for row, col in non_zero_indices:
            self.raw_data.append({"i": row, "k": col, "prob": tot_comat[row][col]})
        self.transform = transform

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        data = None
        if isinstance(idx, (int, slice)):
            data = self.raw_data[idx]
        elif isinstance(idx, (tuple, list)):
            data = [self.raw_data[i] for i in idx]

        if isinstance(idx, (slice, tuple, list)):
            for i in range(len(data)):
                if self.transform:
                    data[i] = self.transform(data[i])
        else:
            if self.transform:
                data[i] = self.transform(data[i])

        return data

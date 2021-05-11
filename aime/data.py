
import os
from functools import partial
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler


class ArrayDict(dict):
    def vmap_(self, fn, rewrite=True):
        for k, v in self.items():
            result = fn(v)
            if rewrite:
                self[k] = result
        return self

    def expand_dim_equal_(
        self, black_list=["image", "frontview_image", "agentview_image"]
    ):
        # TODO: logic is wrong if there is image data in the dict
        max_length = max([len(v.shape) for k, v in self.items() if k not in black_list])
        for k, v in self.items():
            if k in black_list:
                continue
            if len(v.shape) < max_length:
                for _ in range(max_length - len(v.shape)):
                    v = v[..., None]
                self[k] = v
        return self

    def __len__(self) -> int:
        lengths = [len(v) for v in self.values()]
        assert np.all([n == lengths[0] for n in lengths])
        return lengths[0]

    def __getitem__(self, index):
        if isinstance(index, str):
            return dict.__getitem__(self, index)
        else:
            return ArrayDict({k: v[index] for k, v in self.items()})

    def to(self, target: Union[str, torch.Tensor]):
        return self.vmap_(lambda v: v.to(target))

    def to_torch(self):
        return self.vmap_(lambda v: torch.tensor(v))

    def to_numpy(self):
        return self.vmap_(lambda v: v.detach().cpu().numpy())

    def to_float_torch(self):
        return self.vmap_(lambda v: v.float())

    def get_type(self):
        return type(list(self.values())[0])

    @classmethod
    def merge_list(cls, array_dicts: List["ArrayDict"], merge_fn) -> "ArrayDict":
        keys = array_dicts[0].keys()
        return ArrayDict(
            {k: merge_fn([array_dict[k] for array_dict in array_dicts]) for k in keys}
        )

    @classmethod
    def stack(cls, array_dicts: List["ArrayDict"], dim: int) -> "ArrayDict":
        if array_dicts[0].get_type() is torch.Tensor:
            merge_fn = partial(torch.stack, dim=dim)
        else:
            merge_fn = partial(np.stack, axis=dim)
        return cls.merge_list(array_dicts, merge_fn)

    @classmethod
    def cat(cls, array_dicts: List["ArrayDict"], dim: int) -> "ArrayDict":
        if array_dicts[0].get_type() is torch.Tensor:
            merge_fn = partial(torch.cat, dim=dim)
        else:
            merge_fn = partial(np.concatenate, axis=dim)
        return cls.merge_list(array_dicts, merge_fn)


class SequenceDataset(Dataset):
    def __init__(
        self, root: str, horizon: int, overlap: bool, max_capacity: Optional[int] = None
    ) -> None:
        super().__init__()
        self.root = root
        self.horizon = horizon
from __future__ import annotations
from typing import List, Sequence, Tuple, Union
import datetime

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, Dataset


def iter_dataset(ds: Dataset, batch_size: int):
    idx = torch.randperm(len(ds), device=ds.device)
    for i in range(0, len(ds), batch_size):
        yield ds.__getitem__(idx[i: min(i + batch_size, len(ds))])


class MyIterableDataset(IterableDataset):
    def __init__(self, ds: Dataset, batch_size: int):
        self.ds = ds
        self.batch_size = batch_size

    def __iter__(self):
        # return a new generator each time
        return iter_dataset(self.ds, self.batch_size)


class MyDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 float_features: List[str],
                 cat_features: List[str],
                 target_names: Union[str, Sequence[str]],
                 device: str = 'cpu',
                 ):
        self.device = device
        self.float_features = list(float_features)
        self.cat_features = cat_features
        if isinstance(target_names, str):
            self.target_names = [target_names]
        else:
            self.target_names = list(target_names)
        self.number_of_samples = len(df)

        # This version: Assume float features only
#         self.xf = torch.from_numpy(
#             df[self.float_features].values
#         ).to(device=self.device,
#              dtype=torch.float32
#              )
#         self.xc = torch.from_numpy(
#             df[self.cat_features].values
#         ).to(device=self.device,
#              dtype=torch.long
#              )
        self.x = torch.from_numpy(
            df[self.float_features + self.cat_features].values
        ).to(device=self.device,
             dtype=torch.float32
             )
        self.y = torch.from_numpy(
            df[self.target_names].values
        ).to(device=self.device,
             dtype=torch.float32
             )

    def __len__(self) -> int:
        return self.number_of_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.x[idx], self.y[idx]

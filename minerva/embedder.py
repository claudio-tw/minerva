from typing import Sequence, Optional

import torch
from torch import nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self,
                 n_cat_features: int,
                 feat_sizes: Optional[Sequence[int]],
                 emb_dim: int
                 ):
        super().__init__()

        self.embs = nn.ModuleDict()
        self.emb_dim = emb_dim
        self.init_max = []
        if feat_sizes is None:
            assert n_cat_features == 0, ('You need to specify '
                                         'the sizes of each '
                                         'categorical feature')
            feat_sizes = []
        assert len(feat_sizes) == n_cat_features
        for f, size in enumerate(feat_sizes):
            self.embs[str(f)] = nn.Embedding(size+1, emb_dim)
            self.init_max.append(float(
                self.embs[str(f)].weight.detach().abs().max().item()))
        assert len(self.embs) == n_cat_features
        assert len(self.init_max) == n_cat_features
        self.n_cat_features = n_cat_features

    def forward(self, x):
        n, d = x.size()
        out = torch.zeros(
            (n, d, self.emb_dim),
            device=x.device
        )
        for f in range(self.n_cat_features):
            # Add one to account for unknown values (=-1)
            this_emb = self.embs[str(f)](x[:, f].int() + 1)
            # Soft clamping
            out[:, f, :] = F.tanh(0.2*this_emb/self.init_max[f])
        for f in range(self.n_cat_features, d):
            out[:, f, :] = x[:, [f]]
        return out

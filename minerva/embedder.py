from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, feat_sizes: Dict[str, int], emb_dim: int):
        super().__init__()
        self.embs = nn.ModuleDict()
        self.emb_dim = emb_dim
        self.init_max = {}
        for cat, num_features in feat_sizes.items():
            self.embs[cat] = nn.Embedding(num_features+1, emb_dim)
            self.init_max[cat] = float(
                self.embs[cat].weight.detach().abs().max().item())

        self.number_of_embeddings = max(1, len(self.embs))
        self.output_shape = [None, self.number_of_embeddings, emb_dim]

    def forward(self, x):
        out = torch.zeros(
            (len(x), self.number_of_embeddings,  self.emb_dim),
            device=x.device
        )
        for i, (cat, emb) in enumerate(self.embs.items()):
            # Add one to account for unknown values (=-1)
            this_emb = emb.weight[x[:, i] + 1]
            # Soft clamping
            out[:, i, :] = F.tanh(0.2*this_emb/self.init_max[cat])
        return out

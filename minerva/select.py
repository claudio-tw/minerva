from typing import Optional, List, Tuple, Sequence, Dict, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
import torch.nn.functional as F


from .embedder import Embedder
from . import donsker_varadhan


class ResidualBlock(nn.Module):
    def __init__(self, init_dim: int, inner_dim: int, drop_rate: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(init_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, init_dim)
        self.drop1 = nn.Dropout(drop_rate)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        x1 = F.relu(self.drop1(self.fc1(x)))
        x2 = F.relu(self.drop2(self.fc2(x1)))
        tmp = x2.sum()
        assert not torch.isnan(tmp) and not torch.isinf(tmp)
        return x + x2


class ClippingLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_max = self.weight.abs().max().detach()

    def forward(self, input: Tensor) -> Tensor:
        self.init_max = self.init_max.to(input.device)
        w = torch.clamp(self.weight, -2 * self.init_max, 2 * self.init_max)
        return torch.einsum("bn,mn->bm", input, w)


class TestFunction(nn.Sequential):
    def __init__(
        self,
        start_dim: int,
        dim1_mult: int = 16,
        dim1_max: int = 512,
        dim2_mult: int = 8,
        res_blocks: int = 3,
        drop_rate: float = 0.3,
    ):
        d1 = min(start_dim * dim1_mult, dim1_max)
        d2 = dim2_mult * d1
        res = [ResidualBlock(d1, d2, drop_rate=drop_rate)
               for _ in range(res_blocks)]
        super().__init__(ClippingLinear(start_dim, d1), *res, nn.Linear(d1, 1))


class Selector(pl.LightningModule):
    def __init__(
        self,
        cat_features: List[str],
        float_features: List[str],
        targets: List[str],
        mi_threshold: Optional[float] = None,
        lr: float = 1e-3,
        regularization_coef: float = 0.1,
        test_function: Optional[nn.Module] = None,
        dim1_max: int = 256,
        eps: float = 1e-5,
        drift_coef: int = 1,
        num_res_layers: int = 3,
        cat_feat_sizes: Optional[Sequence[int]] = None,
        emb_dim: int = 3,
    ):
        super().__init__()
        self.mi_threshold = mi_threshold
        self.optimal_weights = None
        self.lr = lr
        self.regularization_coef = regularization_coef
        self.feature_names = np.array(cat_features + float_features)
        self.n_features = len(self.feature_names)
        self.n_cat_features = len(cat_features)
        self.n_float_features = len(float_features)
        self.dim_y = len(targets)
        self.emb_dim = emb_dim

        self.dim_joint = emb_dim * self.n_cat_features + \
            self.n_float_features + self.dim_y

        if test_function is None:
            test_function = TestFunction(self.dim_joint,
                                         dim1_max=dim1_max,
                                         res_blocks=num_res_layers)
        self.set_embedder(cat_feat_sizes)

        self.test_function = test_function
        self.eps = eps
        self.drift_coef = drift_coef

        self.enable_projection()

    def set_embedder(self, cat_feat_sizes: Sequence[int]):
        self.embedder = Embedder(
            self.n_cat_features, cat_feat_sizes, self.emb_dim)

    def set_loaders(self, train_dataloader, val_dataloader, test_dataloader):
        self.train_dataloader = lambda: train_dataloader
        self.val_dataloader = lambda: val_dataloader
        self.test_dataloader = lambda: test_dataloader

    def is_projection_enabled(self):
        return self._proj.requires_grad

    def enable_projection(self, weights: Optional[Union[float, np.ndarray, torch.Tensor]] = None):
        # Network convergence is very sensitive to this value.
        # Too high, and it optimizes MI but features don't go sparse
        # Too low, and the opposite happens
        if weights is None:
            if hasattr(self, '_proj'):
                weights = self._proj.clone()
            else:
                weights = torch.ones(self.n_features, requires_grad=False)
        elif isinstance(weights, (float, int)):
            weights = weights * torch.ones(self.n_features)
        elif isinstance(weights, torch.Tensor):
            pass
        elif isinstance(weights, np.ndarray):
            ws = np.array(weights, dtype=np.float32)
            weights = torch.from_numpy(ws)
            assert weights.dtype == torch.float32
        else:
            raise ValueError(weights)

        # one coefficient per feature
        assert len(weights) == self.n_features
        self._proj = nn.Parameter(weights, requires_grad=True)

        self.init_norm = torch.linalg.norm(self._proj).detach().cpu().item()

    def disable_projection(self):
        if hasattr(self, '_proj'):
            ws = self._proj.clone().detach()
        else:
            ws = torch.ones(self.n_features, requires_grad=False)

        self._proj = nn.Parameter(ws,
                                  requires_grad=False)
        self.init_norm = torch.linalg.norm(self._proj).detach().cpu().item()

    def set_projection_from_weights(self, weights: Union[float, Sequence[float], Dict[int, float]], requires_grad: bool = True):
        if isinstance(weights, float):
            ws = [weights] * self.n_features
        else:
            assert len(weights) == self.n_features
            ws = [weights[f] for f in range(len(weights))]
        self._proj = nn.Parameter(torch.Tensor(
            ws), requires_grad=requires_grad)

    def set_projection_from_optimal_weights(self):
        if self.optimal_weights is not None:
            self.set_projection_from_weights(self.optimal_weights)
        else:
            print('No optimal weights have been stored')

    def normalized_proj(self):
        return self._proj / torch.linalg.norm(self._proj)

    def projection_weights(self) -> Dict[int, float]:
        weights = {f: float(self._proj[f])
                   for f in range(len(self._proj))}
        return weights

    def forward(self, x):
        batch_size = x.size(0)
        p = self.normalized_proj()
        z_ = self.embedder(x) * p.reshape(1, -1, 1)
        z = torch.cat([
            z_[:, :self.n_cat_features, :].reshape(batch_size, -1),
            z_[:, self.n_cat_features:, 0]
        ], dim=1
        )
        return z

    def loss_fn(self, z, y):
        mi_loss = self.compute_mi_loss(z, y)
        p = self.normalized_proj()
        regularization = p.abs().sum()
        drift_prevention = torch.linalg.norm(self._proj)
        loss = (
            mi_loss
            + self.regularization_coef * regularization
            + self.drift_coef * (drift_prevention - self.init_norm) ** 2
        )
        components = {"mi_loss": mi_loss, "loss": loss}

        assert not torch.isinf(loss) and not torch.isnan(loss)
        number_of_selected_features = torch.where(
            p.abs() > self.eps, 1, 0).sum().detach()
        self.log("number_of_selected_features",
                 float(number_of_selected_features))
        self.log("normalized_L1", regularization)
        self.log("L2 norm of P", drift_prevention)
        return components

    def selected_feature_names(self):
        p = self.normalized_proj()
        inds = (p.abs() > self.eps).detach().cpu().numpy()
        return self.feature_names[inds]

    def _step(self, batch, batch_idx, plot_name: str):
        x, y = batch
        # embed and multiply by the projection coeffs
        z = self.forward(x)
        loss = self.loss_fn(z, y)
        for k, v in loss.items():
            self.log(plot_name + "_" + k, v)
        return loss["loss"]

    def on_train_epoch_end(self):
        if not hasattr(self, 'weight_history'):
            self.weight_history = []
        weights = self.projection_weights()
        self.weight_history.append(weights)
        if not hasattr(self, 'mutual_info_history'):
            self.mutual_info_history = []
        mi = float(self.val_mutual_information().item())
        self.mutual_info_history.append(mi)
        if self.mi_threshold is not None:
            if self.optimal_weights is None:
                if mi < self.mi_threshold:
                    self.optimal_weights = weights

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "validation")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def compute_mi_loss(self, z, y):
        joint, prod = self._to_mine_joint_and_prod(z, y)
        mi_loss = - donsker_varadhan.v(joint, prod, self.test_function)
        return mi_loss

    def train_mutual_information(self):
        return self.mutual_information(self.train_dataloader())

    def val_mutual_information(self):
        return self.mutual_information(self.val_dataloader())

    def test_mutual_information(self):
        return self.mutual_information(self.test_dataloader())

    @torch.no_grad()
    def mutual_information(self, dataloader):
        device = self._proj.device
        x = dataloader.ds.x.to(device)
        y = dataloader.ds.y.to(device)
        return - self.compute_mi_loss(self.forward(x), y)

    @staticmethod
    def _to_mine_joint_and_prod(
        x: Tensor,
        y: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        joint: Tensor = torch.cat((x, y), dim=1)
        x_: Tensor = x
        x_ = x_[torch.randperm(len(x_)), :]
        prod: Tensor = torch.cat((x_, y), dim=1)
        return joint, prod

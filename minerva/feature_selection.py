from typing import Optional, Any, Dict, Union, List

from pathlib import Path
import numpy as np
import pandas as pd

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


from .iterable_dataset import MyIterableDataset, MyDataset
from .select import Selector
from . import normalize


def dataloaders(
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        float_features: List[str],
        categorical_features: List[str],
        targets: List[str],
        batch_size: int,
):
    dn = normalize.DatasetNormalizer(
        float_cols=float_features + targets,
        categorical_cols=categorical_features,
    )
    train_data = dn.fit_transform(train_data)
    val_data = dn.transform(val_data)
    test_data = dn.transform(test_data)

    train_dataset = MyDataset(
        train_data,
        float_features,
        categorical_features,
        targets
    )
    val_dataset = MyDataset(
        val_data,
        float_features,
        categorical_features,
        targets
    )
    test_dataset = MyDataset(
        test_data,
        float_features,
        categorical_features,
        targets
    )

    train_dataloader = MyIterableDataset(train_dataset, batch_size=batch_size)
    val_dataloader = MyIterableDataset(val_dataset, batch_size=batch_size)
    test_dataloader = MyIterableDataset(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


def train(
        selector_params: Dict[str, Any],
        logger_params: Dict[str, Any],
        train_dataloader: MyIterableDataset,
        val_dataloader: MyIterableDataset,
        test_dataloader: MyIterableDataset,
        reg_coef: float,
        projection_init: Optional[Union[float,
                                        np.ndarray, torch.Tensor]] = None,
        disable_projection: bool = False,
        max_epochs: int = 1000,
        load_path: Optional[Union[str, Path]] = None,
):
    selector_params['regularization_coef'] = reg_coef
    selector = Selector(**selector_params)
    if load_path is not None:
        print(f'loading state dict from {load_path}')
        selector.load_state_dict(torch.load(load_path))

    # Set dataloaders
    selector.set_loaders(train_dataloader, val_dataloader, test_dataloader)

    if projection_init is not None:
        requires_grad = False if disable_projection else True
        selector.set_projection_from_weights(projection_init)

    if disable_projection:
        selector.disable_projection()

    print(f'Regularization coef: {selector.regularization_coef}')
    print(f'Projection enabled: {selector.is_projection_enabled()}')

    # Pre-train projection weights
    weights = {f: round(w, 4)
               for f, w in selector.projection_weights().items()}
    print(f'Pre-train selection weights:\n{weights}\n')

    # Pre-train mutual info
    train_mi = selector.train_mutual_information().item()
    val_mi = selector.val_mutual_information().item()
    print(f'Pre-train train mutual info: {train_mi}')
    print(f'Pre-val val mutual info: {val_mi}')

    # Train the model
    torch.set_float32_matmul_precision("medium")
    logger = TensorBoardLogger("tb_logs", **logger_params)
    trainer = pl.Trainer(
        gradient_clip_val=0.5,
        accelerator="auto",
        log_every_n_steps=50,
        max_epochs=max_epochs,
        logger=logger,
    )
    trainer.fit(
        selector,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    final_test_loss = trainer.test(selector)
    out = final_test_loss[0]
    out["selected_features"] = selector.selected_feature_names()

    # Post-train projection weights
    weights = {f: round(w, 4)
               for f, w in selector.projection_weights().items()}
    print(f'Post-train selection weights:\n{weights}\n')

    # Post-train mutual info
    train_mi = selector.train_mutual_information().item()
    val_mi = selector.val_mutual_information().item()
    print(f'Post-train train mutual info: {train_mi}')
    print(f'Post-val val mutual info: {val_mi}')
    return out, selector


def run(
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        float_features: List[str],
        categorical_features: List[str],
        targets: List[str],
        selector_params: Dict[str, Any],
        logger_params: Dict[str, Any],
        reg_coef: float = 1e5,
        projection_init: float = .25,
        batch_size: int = 500,
        max_epochs: int = 1000,
        model_path: Union[str, Path] = './noreg.pth',
):
    train_dataloader, val_dataloader, test_dataloader = dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        float_features=float_features,
        categorical_features=categorical_features,
        targets=targets,
        batch_size=batch_size,
    )
    out, selector = train(
        selector_params=selector_params,
        logger_params=logger_params,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        reg_coef=.0,
        projection_init=projection_init,
        disable_projection=False,
        max_epochs=max_epochs,
        load_path=None,
    )
    torch.save(selector.state_dict(), model_path)
    out, selector = train(
        selector_params=selector_params,
        logger_params=logger_params,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        reg_coef=reg_coef,
        projection_init=projection_init,
        disable_projection=False,
        max_epochs=max_epochs,
        load_path=model_path,
    )
    # print results
    print(
        'Normalised coefficients of the projection matrix:\n'
        f'{selector.normalized_proj()}\n')
    print(f'Selected features:\n{selector.selected_feature_names()}\n')
    return selector.selected_feature_names(), selector

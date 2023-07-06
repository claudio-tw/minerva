from typing import Optional, Any, Dict, Union, List

from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass

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
    assert train_data.columns.tolist() == val_data.columns.tolist()
    assert train_data.columns.tolist() == test_data.columns.tolist()
    data = pd.concat((train_data, val_data, test_data),
                     axis=0, ignore_index=True)
    data = dn.fit_transform(data)
    train_data = dn.transform(train_data)
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
        learning_rate: float,
        reg_coef: float,
        projection_init: Optional[Union[float,
                                        np.ndarray, torch.Tensor]] = None,
        disable_projection: bool = False,
        max_epochs: int = 1000,
        load_path: Optional[Union[str, Path]] = None,
):
    selector_params['lr'] = learning_rate
    selector_params['regularization_coef'] = reg_coef
    selector = Selector(**selector_params)
    if load_path is not None:
        print(f'loading state dict from {load_path}')
        selector.load_state_dict(torch.load(load_path))
    selector.lr = learning_rate
    selector.regularization_coef = reg_coef

    # Set dataloaders
    selector.set_loaders(train_dataloader, val_dataloader, test_dataloader)

    if projection_init is not None:
        requires_grad = False if disable_projection else True
        selector.set_projection_from_weights(projection_init, requires_grad)

    if disable_projection:
        selector.disable_projection()

    print(f'Learning rate: {selector.lr}')
    print(f'Regularization coef: {selector.regularization_coef}')
    print(f'Projection enabled: {selector.is_projection_enabled()}')

    # Pre-train projection weights
    weights = {f: round(w, 4)
               for f, w in selector.projection_weights().items()}
    print(f'Pre-train selection weights:\n{weights}\n')

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
    return out, selector


@dataclass
class TrainControl:
    number_of_epochs: int
    number_of_segments: int
    data_path: Union[str, Path]
    model_name: str
    learning_rate: float
    reg_coef: float
    projection_init: Optional[Union[float, np.ndarray, torch.Tensor]]
    disable_projection: bool
    first_run_load_path: Optional[Union[str, Path]] = None


def run(
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        float_features: List[str],
        categorical_features: List[str],
        targets: List[str],
        selector_params: Dict[str, Any],
        logger_params: Dict[str, Any],
        noreg_train_control: TrainControl,
        select_train_control: TrainControl,
        batch_size: int
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

    def _run(train_control: TrainControl):
        logs = []
        for segment in range(train_control.number_of_segments):
            print(f'\n----- Segment: {segment}')
            model_path = Path(train_control.data_path) / \
                f'{train_control.model_name}.{segment}'
            if segment == 0:
                load_path = train_control.first_run_load_path
            else:
                load_path = previous_segment_path
            out, selector = train(
                selector_params=selector_params,
                logger_params=logger_params,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                learning_rate=train_control.learning_rate,
                reg_coef=train_control.reg_coef,
                projection_init=train_control.projection_init,
                disable_projection=train_control.disable_projection,
                max_epochs=train_control.number_of_epochs,
                load_path=load_path,
            )
            print(f'Saving state dict to {model_path}')
            torch.save(selector.state_dict(), model_path)
            logs.append(out)
            df_logs = pd.DataFrame(logs)
            logs_path = Path(train_control.data_path) / \
                f'{train_control.model_name}_training_logs.csv'
            df_logs.to_csv(logs_path, index=False)
            weight_history_path = Path(
                train_control.data_path) / \
                f'{train_control.model_name}_weight_history_{segment}.csv'
            weight_history = pd.DataFrame(selector.weight_history)
            weight_history.to_csv(weight_history_path, index=False)
            previous_segment_path = model_path
        return selector, model_path

    print('\n######################################################')
    print(f'No-regularisation training')
    print('######################################################\n')
    _, noreg_model_path = _run(noreg_train_control)

    select_train_control.first_run_load_path = noreg_model_path
    print('\n######################################################')
    print(f'Selection training')
    print('######################################################\n')
    selector, model_path = _run(select_train_control)
    print(f'Selected features:\n{selector.selected_feature_names()}')
    return selector

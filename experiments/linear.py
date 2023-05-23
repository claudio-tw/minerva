#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from minerva import select, normalize
from minerva.iterable_dataset import MyDataset, MyIterableDataset

torch.set_float32_matmul_precision("medium")


# Parameters for the synthetis of data
n = 50000
dx = 10
num_relevant = 2
dy = 1
train_size = int(.66 * n)
val_size = int(.15 * n)
test_size = n - train_size - val_size


# Parameters for training the model
dimension_of_residual_block = 200
lr = 1e-4
num_res_layers = 3
regularization_coef = 1e5
max_epochs = 200
batch_size = 3000


# Synthetise the data
x = np.random.uniform(size=(n, dx))
expected = np.random.choice(dx, replace=False, size=num_relevant)
y = (
    np.random.uniform(size=(1, dy, num_relevant)
                      ) @ np.expand_dims(x[:, expected], axis=2)
)[:, :, 0]
feature_cols = [f'f{n}' for n in range(dx)]
float_features = feature_cols
cat_features = []
target_cols = [f'y{n}' for n in range(dy)]
target_names = target_cols
xdf = pd.DataFrame(
    x,
    columns=feature_cols
)
ydf = pd.DataFrame(
    y,
    columns=target_cols
)
data = pd.concat((xdf, ydf), axis=1)
train_data = data.iloc[:train_size]
val_data = data.iloc[train_size: train_size + val_size]
test_data = data.iloc[train_size + val_size:]


# Prepare the data for the training
dn = normalize.DatasetNormalizer(
    float_cols=feature_cols + target_cols, categorical_cols=[])
train_data = dn.fit_transform(train_data)
val_data = dn.transform(val_data)
test_data = dn.transform(test_data)

train_dataset = MyDataset(
    train_data,
    float_features,
    cat_features,
    target_names
)
val_dataset = MyDataset(
    val_data,
    float_features,
    cat_features,
    target_names
)
test_dataset = MyDataset(
    test_data,
    float_features,
    cat_features,
    target_names
)

train_dataloader = MyIterableDataset(train_dataset, batch_size=batch_size)
val_dataloader = MyIterableDataset(val_dataset, batch_size=batch_size)
test_dataloader = MyIterableDataset(test_dataset, batch_size=batch_size)


# Instantiate the model
selector = select.Selector(
    feature_cols=feature_cols,
    target_cols=target_names,
    dim1_max=dimension_of_residual_block,
    lr=lr,
    num_res_layers=num_res_layers,
    regularization_coef=regularization_coef,
)

selector.set_loaders(train_dataloader, val_dataloader, test_dataloader)

# Train the model
logger = TensorBoardLogger("tb_logs", name='linear')
trainer = pl.Trainer(
    gradient_clip_val=.5,
    accelerator="auto",
    log_every_n_steps=10,
    max_epochs=max_epochs,
    logger=logger
)
trainer.fit(
    selector,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)

# print results
print(
    f'Normalised coefficients of the projection matrix:\n{selector.normalized_proj()}\n')
print(f'Selected features:\n{selector.selected_feature_names()}\n')
print(f'Expected features\n{expected}\n')

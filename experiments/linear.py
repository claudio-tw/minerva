#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import pandas as pd
import numpy as np

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


from minerva.select import Selector
from minerva.iterable_dataset import MyDataset, MyIterableDataset
from minerva import normalize


torch.set_float32_matmul_precision("medium")


pth = Path('./data/linear')
pth.mkdir(exist_ok=True, parents=True)


# Parameters for the synthetis of data
n = 50000
dx = 10
num_relevant = 2
dy = 1
train_size = int(.66 * n)
val_size = int(.15 * n)
test_size = n - train_size - val_size


# Set metaparameters

num_samples = n
# The below makes things quite slow; 256 and 3 seem to perform almost as well, but way faster
dimension_of_residual_block = 512
num_res_layers = 4
scaler = 2  # Scaler = 4 did the best so far, scaler=8 diverged
batch_size = scaler*2048
num_batches = num_samples // batch_size
max_epochs = int(2000*scaler)  # to keep the number of batches constant

lr = 1e-5  # scaling that as sqrt(scaler) didn't seem to work


# Synthesize the data
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


def run_this(reg_coef: float, load_path=None, wgt_mult=None):

    selector = Selector(
        feature_cols=feature_cols,
        target_cols=target_names,
        dim1_max=dimension_of_residual_block,
        lr=lr,
        num_res_layers=num_res_layers,
        regularization_coef=reg_coef,
        eps=.001
    )
    if load_path is not None:
        selector.load_state_dict(torch.load(load_path))

    # Set dataloaders
    selector.set_loaders(train_dataloader, val_dataloader, test_dataloader)

    selector.enable_projection(wgt_mult=wgt_mult)

    # Train the model
    logger = TensorBoardLogger("tb_logs", name="linear")
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
    return out, selector


noreg_path = "./data/linear/noreg_small.pth"


# Train a long run without reg, to get the MI network right
out, selector = run_this(reg_coef=0.0, wgt_mult=None)
torch.save(selector.state_dict(), noreg_path)


# now add reg starting from that snapshot
# Regularization level appears to have almost no effect as long as it's > 100
reg_coefs = [1e5]
results = []
for reg_coef in reg_coefs:
    out, selector = run_this(
        reg_coef=reg_coef, load_path=noreg_path, wgt_mult=.25)
    results.append(out)
    results[-1]["reg_coef"] = reg_coef
    df = pd.DataFrame(results)
    df.to_csv("./data/linear/results.csv")


# print results
print(
    f'Normalised coefficients of the projection matrix:\n{selector.normalized_proj()}\n')
print(f'Selected features:\n{selector.selected_feature_names()}\n')
print(f'Expected features:\n{expected}\n')


print(
    f'Mutual information on train dataset: {float(selector.train_mutual_information())}')
print(
    f'Mutual information on val dataset: {float(selector.val_mutual_information())}')
print(
    f'Mutual information on test dataset: {float(selector.test_mutual_information())}')

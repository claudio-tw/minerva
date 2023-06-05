#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt


from minerva import select
from minerva.iterable_dataset import MyDataset, MyIterableDataset


def sample(corr: float, number_of_samples: int) -> pd.DataFrame:
    cov = np.array([[1., corr], [corr, 1.]])
    mean = np.array([.0, .0])
    z = multivariate_normal.rvs(
        mean=mean,
        cov=cov,
        size=number_of_samples
    )
    df = pd.DataFrame(z, columns=['x', 'y'])
    return df


def run(
    corr: float,
    number_of_samples: int = 100000,
    dimension_of_residual_block: int = 200,
    lr: float = 1e-4,
    num_res_layers: int = 3,
    max_epochs: int = 300,
    batch_size: int = 2000,
):
    n = number_of_samples
    train_size = int(.66 * n)
    val_size = int(.15 * n)
    test_size = n - train_size - val_size  # NOQA
    feature_cols = ['x']
    float_features = feature_cols
    targets = ['y']
    cat_features = []
    exact_mi = - .5 * np.log(1. - corr * corr)
    data = sample(corr, number_of_samples)
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size: train_size + val_size]
    test_data = data.iloc[train_size + val_size:]
    train_dataset = MyDataset(
        train_data,
        float_features,
        cat_features,
        targets
    )
    val_dataset = MyDataset(
        val_data,
        float_features,
        cat_features,
        targets
    )
    test_dataset = MyDataset(
        test_data,
        float_features,
        cat_features,
        targets
    )
    train_dataloader = MyIterableDataset(train_dataset, batch_size=batch_size)
    val_dataloader = MyIterableDataset(val_dataset, batch_size=batch_size)
    test_dataloader = MyIterableDataset(test_dataset, batch_size=batch_size)
    selector = select.Selector(
        float_features=float_features,
        cat_features=[],
        targets=targets,
        dim1_max=dimension_of_residual_block,
        lr=lr,
        num_res_layers=num_res_layers,
        regularization_coef=0.,
        drift_coef=.0,
    )
    selector.set_embedder([], 1)
    selector.disable_projection()
    selector.set_loaders(train_dataloader, val_dataloader, test_dataloader)
    logger = TensorBoardLogger("tb_logs", name='normalsmile')
    trainer = pl.Trainer(
        gradient_clip_val=.5,
        accelerator="auto",
        log_every_n_steps=10,
        max_epochs=max_epochs,
        logger=logger,
    )
    trainer.fit(
        selector,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    train_mi = float(selector.train_mutual_information())
    val_mi = float(selector.val_mutual_information())
    test_mi = float(selector.test_mutual_information())
    print('\n\n#########################################################################')
    print(f'Correlation: {corr}')
    print(f'Exact mutual information: {exact_mi}')
    print(f'Mutual information on train dataset: {train_mi}')
    print(f'Mutual information on val dataset: {val_mi}')
    print(f'Mutual information on test dataset: {test_mi}')
    res = dict(
        exact_mi=exact_mi,
        train_mi=train_mi,
        val_mi=val_mi,
        test_mi=test_mi
    )
    return res


# Hyperparameters of the experiment
rhos = np.concatenate((np.linspace(-.95, -.01, num=5),
                      np.linspace(.01, .95, num=5)))
number_of_samples = 100000

dimension_of_residual_block = 100
lr = 3e-4
num_res_layers = 3
max_epochs = 100
batch_size = 4000

# Run the experiment
exact_mis = []
train_mis = []
val_mis = []
test_mis = []
for rho in rhos:
    res = run(
        corr=rho,
        number_of_samples=number_of_samples,
        dimension_of_residual_block=dimension_of_residual_block,
        lr=lr,
        num_res_layers=num_res_layers,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    train_mis.append(res['train_mi'])
    val_mis.append(res['val_mi'])
    test_mis.append(res['test_mi'])
    exact_mis.append(res['exact_mi'])

# Collect results
df = pd.DataFrame(
    {
        'exact_mi': exact_mis,
        'train_mi': train_mis,
        'val_mi': val_mis,
        'test_mi': test_mis
    },
    index=rhos
)


# Plot results
fig = plt.figure()
ax = fig.add_subplot()
rhos = np.linspace(-.95, .95, num=100)
exact_mis = - .5 * np.log(1. - rhos * rhos)
ax.plot(rhos, exact_mis, label='exact_mi')
ax = df[['train_mi', 'test_mi']].plot(
    style={
        'train_mi': '*--',
        'test_mi': '*--'
    },
    ax=ax
)
fig.legend()
plt.show()

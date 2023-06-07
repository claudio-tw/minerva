#!/usr/bin/env python
# coding: utf-8

from typing import Sequence
from pathlib import Path
import pandas as pd
import numpy as np

from minerva import feature_selection


pth = Path('./data/categorical')
pth.mkdir(exist_ok=True, parents=True)


# Parameters for the synthetis of data
n = 50000
dx = 10
num_relevant = 2
cat_feat_sizes = np.random.randint(low=7, high=10, size=(dx))
dy = 1
feature_cols = [f'f{n}' for n in range(dx)]
float_features = []  # no feature is float
cat_features = feature_cols  # all features are categorical
targets = [f'y{n}' for n in range(dy)]
train_size = int(.66 * n)
val_size = int(.15 * n)
test_size = n - train_size - val_size


# Metaparameters
num_samples = n
dimension_of_residual_block = 512
num_res_layers = 4
scaler = 2  # Scaler = 4 did the best so far, scaler=8 diverged
batch_size = scaler*2048
num_batches = num_samples // batch_size
max_epochs = int(2000*scaler)  # to keep the number of batches constant
lr = 1e-5  # scaling that as sqrt(scaler) didn't seem to work
emb_dim = 3
reg_coef = 1e5

model_path = "./data/categorical/noreg.pth"

# Pack hyperparameters
selector_params = dict(
    cat_features=cat_features,
    float_features=float_features,
    targets=targets,
    dim1_max=dimension_of_residual_block,
    lr=lr,
    num_res_layers=num_res_layers,
    eps=.001,
    cat_feat_sizes=cat_feat_sizes,
    emb_dim=emb_dim,
)
logger_params = dict(
    name="categorical"
)


def synthesize_data(
    n: int,
    dx: int,
    num_relevant: int,
    feat_sizes: Sequence[int],
    dy: int,
    train_size: int,
    val_size: int,
    test_size: int,
):
    xs = [
        np.random.randint(low=0, high=size, size=(n, 1))
        for size in feat_sizes
    ]
    x = np.concatenate(xs, axis=1)
    expected = np.random.choice(dx, replace=False, size=num_relevant)
    y = np.zeros(shape=(n,), dtype=int)
    for f0, f1 in zip(expected[:-1], expected[1:]):
        x0 = x[:, f0] / feat_sizes[f0]
        x1 = x[:, f1] / feat_sizes[f1]
        y += np.array(x0 > x1, dtype=int)

    feature_cols = [f'f{n}' for n in range(dx)]
    targets = [f'y{n}' for n in range(dy)]
    xdf = pd.DataFrame(
        x,
        columns=feature_cols
    )
    ydf = pd.DataFrame(
        y,
        columns=targets
    )
    data = pd.concat((xdf, ydf), axis=1)
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size: train_size + val_size]
    test_data = data.iloc[train_size + val_size:]

    return expected, train_data, val_data, test_data


def main():

    # Synthesize the data
    expected_features, train_data, val_data, test_data = synthesize_data(
        n=n,
        dx=dx,
        num_relevant=num_relevant,
        feat_sizes=cat_feat_sizes,
        dy=dy,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    # Run feature selection
    selected_features, selector = feature_selection.run(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        float_features=float_features,
        categorical_features=cat_features,
        targets=targets,
        selector_params=selector_params,
        logger_params=logger_params,
        reg_coef=reg_coef,
        projection_init=.25,
        batch_size=batch_size,
        max_epochs=max_epochs,
        model_path=model_path,
    )

    # print results
    print(
        f'Normalised coefficients of the projection matrix:\n{selector.normalized_proj()}\n')
    print(f'Selected features:\n{selector.selected_feature_names()}\n')
    print(f'Expected features:\n{expected_features}\n')

    print(
        f'Mutual information on train dataset: {float(selector.train_mutual_information())}')
    print(
        f'Mutual information on val dataset: {float(selector.val_mutual_information())}')
    print(
        f'Mutual information on test dataset: {float(selector.test_mutual_information())}')


if __name__ == '__main__':
    main()

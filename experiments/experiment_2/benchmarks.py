#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import torch

import tools


def main():
    n = 50000
    dy = 1
    num_cat_features = 10
    num_cont_features = 30
    feature_cols = [f'x{n}' for n in range(
        num_cat_features + num_cont_features)]
    cat_features = feature_cols[:num_cat_features]
    float_features = feature_cols[num_cat_features:]
    targets = [f'y{n}' for n in range(dy)]

    data = pd.read_csv('data/large.csv')
    xdf = data.loc[:, feature_cols]
    x = xdf.values
    ydf = data.loc[:, targets]
    y = ydf.values
    store = pickle.load(open('data/store.exp2', 'rb'))

    expected_cat = store['expected_cat']
    expected_cont0 = store['expected_cont0']
    expected_cont1 = store['expected_cont1']
    expected_cont = store['expected_cont']
    expected_features = store['expected_features']

    # ### Uncover relation between features and data
    _chooser = data.iloc[:, expected_cat[1]] == data.iloc[:, expected_cat[0]]
    idx0 = _chooser == 0
    idx1 = _chooser == 1
    y_ = np.zeros(shape=(len(data), dy))
    y_[idx0, :] = (
        store['t0'] @ np.expand_dims(
            np.sin(2 * np.pi * data.loc[idx0].iloc[:, expected_cont0]),
            axis=2))[:, :, 0]
    y_[idx1, :] = (
        store['t1'] @ np.expand_dims(
            np.cos(2 * np.pi * data.loc[idx1].iloc[:, expected_cont1]),
            axis=2))[:, :, 0]

    assert np.allclose(np.squeeze(y_), data['y0'].values, atol=1e-6, rtol=1e-4)

    # ### Selection with marginal 1D ksg mutual info
    ksgselection, mis = tools.ksgmi(xdf, ydf, threshold=0.01)
    print(f'Expected features: {sorted(expected_features)}')
    print(f'Marginal KSG selection: {sorted(ksgselection)}')

    # ### Selection with HSIC Lasso
    xfeattype = tools.FeatureType.FLOAT
    yfeattype = tools.FeatureType.FLOAT
    hsiclasso_selection = tools.pyhsiclasso(
        x, y, xfeattype=xfeattype, yfeattype=yfeattype, n_features=10, batch_size=500)
    print(f'Expected features: {sorted(expected_features)}')
    print(f'HSIC Lasso selection: {sorted(hsiclasso_selection)}')


if __name__ == '__main__':
    main()

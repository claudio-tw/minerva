#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import pickle
import pandas as pd
import numpy as np


import tools
from experiment_2 import utils


def main():
    xdf, ydf, float_features, cat_features, targets = utils.load_data(
        'data/exp2large.csv')
    all_features = cat_features + float_features
    num_samples = 500000
    xdf = xdf.iloc[:num_samples]
    ydf = ydf.iloc[:num_samples]
    x = xdf.values
    y = ydf.values

    # ### Selection with marginal 1D ksg mutual info
    print(f'\n\n###########################################################')
    print('KSG')
    print(f'###########################################################\n')
    ksgselection, mis = tools.ksgmi(xdf, ydf, threshold=0.02)
    ksgselection = list(np.array(all_features)[sorted(ksgselection)])
    print(f'Marginal KSG selection: {ksgselection}')
    del ksgselection
    del mis

    # ### Selection with HSIC Lasso
    print(f'\n\n###########################################################')
    print('HSIC Lasso')
    print(f'###########################################################\n')
    xfeattype = tools.FeatureType.FLOAT
    yfeattype = tools.FeatureType.FLOAT
    hsiclasso_selection = set()
    batch_size = 3000
    num_batches = len(x) // batch_size
    for n in range(num_batches):
        print(f'Batch number: {n}')
        t0 = n * batch_size
        t1 = (n + 1) * batch_size
        sel = tools.pyhsiclasso(
            x[t0:t1, :], y[t0:t1, :],
            xfeattype=xfeattype,
            yfeattype=yfeattype,
            n_features=30,
            batch_size=300)
        print(sel)
        hsiclasso_selection = hsiclasso_selection.union(set(sel))
    hsiclasso_selection = list(
        np.array(all_features)[sorted(list(hsiclasso_selection))]
    )
    print(f'HSIC Lasso selection: {hsiclasso_selection}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import torch
from arfs.feature_selection import allrelevant
from arfs.feature_selection.allrelevant import Leshy
from xgboost import XGBRegressor

import tools


def main():
    xdf, ydf, float_features, cat_features = tools.load_data()
    all_features = cat_features + float_features
    x = xdf.values
    y = ydf.values

    # ### Selection with marginal 1D ksg mutual info
    ksgselection, mis = tools.ksgmi(xdf, ydf, threshold=0.02)
    ksgselection = list(np.array(all_features)[sorted(ksgselection)])
    print(f'Marginal KSG selection: {ksgselection}')

    # ### Selection with HSIC Lasso
    xfeattype = tools.FeatureType.FLOAT
    yfeattype = tools.FeatureType.FLOAT
    hsiclasso_selection = set()
    for n in range(2):
        t0 = n * 5000
        t1 = (n+1) * 5000
        sel = tools.pyhsiclasso(
            x[t0:t1, :], y[t0:t1, :],
            xfeattype=xfeattype,
            yfeattype=yfeattype,
            n_features=10,
            batch_size=500)
        print(sel)
        hsiclasso_selection = hsiclasso_selection.union(set(sel))
    hsiclasso_selection = list(
        np.array(all_features)[sorted(list(hsiclasso_selection))]
    )
    print(f'HSIC Lasso selection: {hsiclasso_selection}')

    # ### Selection with Boruta
    n_estimators = 'auto'
    importance = "native"
    max_iter = 100
    random_state = None
    verbose = 0
    keep_weak = False
    yser = ydf['TRANSFER3M_TARGET']
    regressor = XGBRegressor(random_state=42)
    leshy = Leshy(
        regressor,
        n_estimators=n_estimators,
        importance=importance,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
        keep_weak=keep_weak,
    )
    leshy.fit(xdf, yser)
    leshy_selection = list(leshy.selected_features_)
    print(f'Boruta selection: {leshy_selection}')


if __name__ == '__main__':
    main()

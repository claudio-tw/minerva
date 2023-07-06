#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import pandas as pd
import numpy as np
import torch
from arfs.feature_selection import allrelevant
from arfs.feature_selection.allrelevant import Leshy
from xgboost import XGBRegressor

from experiment_2 import utils


def main():
    xdf, ydf, float_features, cat_features, targets = utils.load_data(
        'data/exp2.csv')
    all_features = cat_features + float_features
    x = xdf.values
    y = ydf.values

    # ### Selection with Boruta
    n_estimators = 'auto'
    importance = "native"
    max_iter = 100
    random_state = None
    verbose = 0
    keep_weak = False
    yser = ydf['y']
    random_state = np.random.randint(low=0, high=100)
    regressor = XGBRegressor(random_state=random_state)
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
    print(f'Random state used in XGBRegressor: {random_state}')
    print(f'Number of selected features: {len(leshy_selection)}')
    print(f'Boruta selection: {leshy_selection}')


if __name__ == '__main__':
    main()

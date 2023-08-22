#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import pandas as pd
from arfs.feature_selection import allrelevant
from arfs.feature_selection.allrelevant import Leshy
from xgboost import XGBRegressor

from experiment_3 import utils


def main(batch_size=100000):
    xdf, ydf, float_features, cat_features, targets = utils.load_data(
        'data/exp3.csv')
    all_features = cat_features + float_features

    # ### Selection with Boruta
    n_estimators = 'auto'
    importance = "native"
    max_iter = 100
    random_state = None
    verbose = 0
    keep_weak = False
    yser = ydf['y']
    boruta_selection = set()
    batch_size = min(batch_size, len(xdf))
    for n in range(batch_size, len(xdf), batch_size):
        print(f'Batch number: {n//batch_size}')
        t0 = n
        t1 = n + batch_size
        if t1 >= len(xdf) - 1:
            break
        predictors = xdf.iloc[t0:t1, :]
        ys = yser.iloc[t0:t1]
        regressor = XGBRegressor(random_state=random_state)
        leshy = Leshy(
            regressor,
            n_estimators=n_estimators,
            importance=importance,
            max_iter=max_iter,
            random_state=None,
            verbose=verbose,
            keep_weak=keep_weak,
        )
        leshy.fit(predictors, ys)
        leshy_selection = list(leshy.selected_features_)
        print(f'Random state used in XGBRegressor: {random_state}')
        print(f'Number of selected features: {len(leshy_selection)}')
        print(f'Boruta partial selection: {leshy_selection}')
        boruta_selection = boruta_selection.union(set(leshy_selection))
    boruta_selection = sorted(list(boruta_selection))
    print(f'Boruta overall selection: {boruta_selection}')


if __name__ == '__main__':
    main()

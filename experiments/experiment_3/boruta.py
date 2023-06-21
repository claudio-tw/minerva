#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
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

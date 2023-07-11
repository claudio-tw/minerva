#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import torch
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from experiment_2 import utils
from data.benchmark_selection_filtered import (
    ksg_selection,
    hsic_selection,
    boruta_selection,
    hisel_selection,
)
from data.minerva_selection import (
    minerva_selection_1,
    minerva_filtered_selection_1,
    minerva_selection_1_adjusted,
)

# ### CatBoost parameters
random_state = np.random.randint(low=0, high=100)
print(f'Random state used in CatBoostRegressor: {random_state}')
catboost_params = {
    "iterations": 10000,  # Early stopping will reduce this number of iterations
    "depth": 8,
    'random_state': random_state,
    'verbose': False
}


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, selection=None):
    if selection is None:
        selection = X_train.columns.tolist()
    x_train = X_train.loc[:, selection].copy()
    x_val = X_val.loc[:, selection].copy()
    x_test = X_test.loc[:, selection].copy()
    model = CatBoostRegressor(**catboost_params)
    model.fit(x_train, y_train, eval_set=(
        x_val, y_val), early_stopping_rounds=20)
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    r2_insample = r2_score(y_true=y_train, y_pred=train_predictions)
    r2_outsample = r2_score(y_true=y_test, y_pred=test_predictions)
    return r2_insample, r2_outsample


def main(dataset_path='data/exp2filtered.csv'):  # or 'data/exp2.csv'):
    xdf, ydf, float_features, cat_features, targets = utils.load_data(
        dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        xdf,
        ydf,
        test_size=0.2,
        random_state=None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        xdf,
        ydf,
        test_size=0.3,
        random_state=None
    )

    # ## Accuracy of prediction based on all features
    r2_insample, r2_outsample = train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        selection=None
    )
    print('\nAll features - No selection')
    print(f'Number of features: {X_train.shape[1]}')
    print(f'In-sample R2 score: {round(r2_insample, 4)}')
    print(f'Out-sample R2 score: {round(r2_outsample, 4)}')

    # ## Accuracy of prediction based on KSG selection
    r2_insample_ksg, r2_outsample_ksg = train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        selection=ksg_selection,
    )
    print('\nKSG')
    print(f'Number of features: {len(ksg_selection)}')
    print(
        f'In-sample R2 score with KSG selection: {round(r2_insample_ksg, 4)}')
    print(
        f'Out-sample R2 score with KSG selection: {round(r2_outsample_ksg, 4)}')

    # ## Accuracy of prediction based on HSIC selection
    r2_insample_hsic, r2_outsample_hsic = train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        selection=hsic_selection,
    )
    print('\nHSIC Lasso')
    print(f'Number of features: {len(hsic_selection)}')
    print(
        f'In-sample R2 score with HSIC Lasso selection: {round(r2_insample_hsic, 4)}')
    print(
        f'Out-sample R2 score with HSIC Lasso selection: {round(r2_outsample_hsic, 4)}')

    # ## Accuracy of prediction based on HISEL selection
    r2_insample_hisel, r2_outsample_hisel = train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        selection=hisel_selection,
    )
    print('\nHISEL')
    print(f'Number of features: {len(hisel_selection)}')
    print(
        f'In-sample R2 score with HISEL selection: {round(r2_insample_hisel, 4)}')
    print(
        f'Out-sample R2 score with HISEL selection: {round(r2_outsample_hisel, 4)}')

    # ## Accuracy of prediction based on boruta selection
    r2_insample_boruta, r2_outsample_boruta = train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        selection=boruta_selection,
    )
    print('\nBORUTA')
    print(f'Number of features: {len(boruta_selection)}')
    print(
        f'In-sample R2 score with Boruta selection: {round(r2_insample_boruta, 4)}')
    print(
        f'Out-sample R2 score with Boruta selection: {round(r2_outsample_boruta, 4)}')

    # ## Accuracy of prediction based on minerva selection
    r2_insample_minerva, r2_outsample_minerva = train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        selection=minerva_filtered_selection_1,
    )
    print('\nMINERVA')
    print(f'Number of features: {len(minerva_filtered_selection_1)}')
    print(
        f'In-sample R2 score with Minerva selection: {round(r2_insample_minerva, 4)}')
    print(
        f'Out-sample R2 score with Minerva selection: {round(r2_outsample_minerva, 4)}')


if __name__ == '__main__':
    main()

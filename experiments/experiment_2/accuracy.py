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


def main():
    xdf, ydf, float_features, cat_features, targets = utils.load_data(
        'data/exp2filtered.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        xdf,
        ydf,
        test_size=0.3,
        random_state=40
    )
    # ### CatBoost parameters
    # random_state = 15 gives good R2 scores for Minerva selection 1
    # random_state = 3 or 9 gives good R2 scores for adjusted Minerva selection 1
    # random_state = 1 gives bad R2 scores for Boruta and good R2 scores for Minerva selection 1
    random_state = np.random.randint(low=0, high=100)
    print(f'Random state used in CatBoostRegressor: {random_state}')
    params = {
        "iterations": 200,
        "depth": 8,
        'random_state': random_state,
        'verbose': False
    }

    # ## Accuracy of prediction based on all features
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    r2_insample = r2_score(y_true=y_train, y_pred=train_predictions)
    r2_outsample = r2_score(y_true=y_test, y_pred=test_predictions)
    print('\nAll features - No selection')
    print(f'Number of features: {X_train.shape[1]}')
    print(f'In-sample R2 score: {round(r2_insample, 4)}')
    print(f'Out-sample R2 score: {round(r2_outsample, 4)}')

    # ## Accuracy of prediction based on KSG selection
    X_train_ksg = X_train.loc[:, ksg_selection]
    X_test_ksg = X_test.loc[:, ksg_selection]
    model_ksg = CatBoostRegressor(**params)
    model_ksg.fit(X_train_ksg, y_train)
    train_predictions_ksg = model_ksg.predict(X_train_ksg)
    test_predictions_ksg = model_ksg.predict(X_test_ksg)
    r2_insample_ksg = r2_score(y_true=y_train, y_pred=train_predictions_ksg)
    r2_outsample_ksg = r2_score(y_true=y_test, y_pred=test_predictions_ksg)
    print('\nKSG')
    print(f'Number of features: {len(ksg_selection)}')
    print(
        f'In-sample R2 score with KSG selection: {round(r2_insample_ksg, 4)}')
    print(
        f'Out-sample R2 score with KSG selection: {round(r2_outsample_ksg, 4)}')

    # ## Accuracy of prediction based on HSIC selection
    X_train_hsic = X_train.loc[:, hsic_selection]
    X_test_hsic = X_test.loc[:, hsic_selection]
    model_hsic = CatBoostRegressor(**params)
    model_hsic.fit(X_train_hsic, y_train)
    train_predictions_hsic = model_hsic.predict(X_train_hsic)
    test_predictions_hsic = model_hsic.predict(X_test_hsic)
    r2_insample_hsic = r2_score(y_true=y_train, y_pred=train_predictions_hsic)
    r2_outsample_hsic = r2_score(y_true=y_test, y_pred=test_predictions_hsic)
    print('\nHSIC Lasso')
    print(f'Number of features: {len(hsic_selection)}')
    print(
        f'In-sample R2 score with HSIC Lasso selection: {round(r2_insample_hsic, 4)}')
    print(
        f'Out-sample R2 score with HSIC Lasso selection: {round(r2_outsample_hsic, 4)}')

    # ## Accuracy of prediction based on HISEL selection
    X_train_hisel = X_train.loc[:, hisel_selection]
    X_test_hisel = X_test.loc[:, hisel_selection]
    model_hisel = CatBoostRegressor(**params)
    model_hisel.fit(X_train_hisel, y_train)
    train_predictions_hisel = model_hisel.predict(X_train_hisel)
    test_predictions_hisel = model_hisel.predict(X_test_hisel)
    r2_insample_hisel = r2_score(
        y_true=y_train, y_pred=train_predictions_hisel)
    r2_outsample_hisel = r2_score(y_true=y_test, y_pred=test_predictions_hisel)
    print('\nHISEL Lasso')
    print(f'Number of features: {len(hisel_selection)}')
    print(
        f'In-sample R2 score with HISEL Lasso selection: {round(r2_insample_hisel, 4)}')
    print(
        f'Out-sample R2 score with HISEL Lasso selection: {round(r2_outsample_hisel, 4)}')

    # ## Accuracy of prediction based on boruta selection
    X_train_boruta = X_train.loc[:, boruta_selection]
    X_test_boruta = X_test.loc[:, boruta_selection]
    model_boruta = CatBoostRegressor(**params)
    model_boruta.fit(X_train_boruta, y_train)
    train_predictions_boruta = model_boruta.predict(X_train_boruta)
    test_predictions_boruta = model_boruta.predict(X_test_boruta)
    r2_insample_boruta = r2_score(
        y_true=y_train, y_pred=train_predictions_boruta)
    r2_outsample_boruta = r2_score(
        y_true=y_test, y_pred=test_predictions_boruta)
    print('\nBORUTA')
    print(f'Number of features: {len(boruta_selection)}')
    print(
        f'In-sample R2 score with Boruta selection: {round(r2_insample_boruta, 4)}')
    print(
        f'Out-sample R2 score with Boruta selection: {round(r2_outsample_boruta, 4)}')

    # ## Accuracy of prediction based on minerva selection
    X_train_minerva = X_train.loc[:, minerva_filtered_selection_1]
    X_test_minerva = X_test.loc[:, minerva_filtered_selection_1]
    model_minerva = CatBoostRegressor(**params)
    model_minerva.fit(X_train_minerva, y_train)
    train_predictions_minerva = model_minerva.predict(X_train_minerva)
    test_predictions_minerva = model_minerva.predict(X_test_minerva)
    r2_insample_minerva = r2_score(
        y_true=y_train, y_pred=train_predictions_minerva)
    r2_outsample_minerva = r2_score(
        y_true=y_test, y_pred=test_predictions_minerva)
    print('\nMINERVA')
    print(f'Number of features: {len(minerva_filtered_selection_1)}')
    print(
        f'In-sample R2 score with Minerva selection: {round(r2_insample_minerva, 4)}')
    print(
        f'Out-sample R2 score with Minerva selection: {round(r2_outsample_minerva, 4)}')
    X_train_minerva = X_train.loc[:, minerva_selection_1_adjusted]
    X_test_minerva = X_test.loc[:, minerva_selection_1_adjusted]
    model_minerva = CatBoostRegressor(**params)
    model_minerva.fit(X_train_minerva, y_train)
    train_predictions_minerva = model_minerva.predict(X_train_minerva)
    test_predictions_minerva = model_minerva.predict(X_test_minerva)
    r2_insample_minerva = r2_score(
        y_true=y_train, y_pred=train_predictions_minerva)
    r2_outsample_minerva = r2_score(
        y_true=y_test, y_pred=test_predictions_minerva)
    print(
        f'Number of features in adjusted selection: {len(minerva_selection_1_adjusted)}')
    print(
        f'In-sample R2 score with Minerva adjusted selection: {round(r2_insample_minerva, 4)}')
    print(
        f'Out-sample R2 score with Minerva adjusted selection: {round(r2_outsample_minerva, 4)}')


if __name__ == '__main__':
    main()

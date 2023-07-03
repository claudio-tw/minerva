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

import tools
from data.benchmark_selection import ksg_selection, hsic_selection, boruta_selection
from data.minerva_selection import minerva_selection


def main():
    n = 100000
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

    X_train, X_test, y_train, y_test = train_test_split(
        xdf,
        ydf,
        test_size=0.3,
        random_state=40
    )
    # ### CatBoost parameters
    params = {
        "iterations": 150,
        "depth": 8,
        "verbose": 10,
        'random_state': 40,
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
    print(
        f'In-sample R2 score with HSIC Lasso selection: {round(r2_insample_hsic, 4)}')
    print(
        f'Out-sample R2 score with HSIC Lasso selection: {round(r2_outsample_hsic, 4)}')

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
    print(
        f'In-sample R2 score with Boruta selection: {round(r2_insample_boruta, 4)}')
    print(
        f'Out-sample R2 score with Boruta selection: {round(r2_outsample_boruta, 4)}')

    # ## Accuracy of prediction based on minerva selection
    X_train_minerva = X_train.loc[:, minerva_selection]
    X_test_minerva = X_test.loc[:, minerva_selection]
    model_minerva = CatBoostRegressor(**params)
    model_minerva.fit(X_train_minerva, y_train)
    train_predictions_minerva = model_minerva.predict(X_train_minerva)
    test_predictions_minerva = model_minerva.predict(X_test_minerva)
    r2_insample_minerva = r2_score(
        y_true=y_train, y_pred=train_predictions_minerva)
    r2_outsample_minerva = r2_score(
        y_true=y_test, y_pred=test_predictions_minerva)
    print('\nMINERVA')
    print(
        f'In-sample R2 score with minerva selection: {round(r2_insample_minerva, 4)}')
    print(
        f'Out-sample R2 score with minerva selection: {round(r2_outsample_minerva, 4)}')


if __name__ == '__main__':
    main()

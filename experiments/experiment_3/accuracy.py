#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve

from experiment_3 import utils
from data.benchmark_selection import (
    ksg_selection,
    #     hsic_selection,
    #     boruta_selection,
    #     hisel_selection,
)
from data.minerva_selection import (
    minerva_selection_1,
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
    model = CatBoostClassifier(**catboost_params)
    model.fit(x_train, y_train, eval_set=(
        x_val, y_val), early_stopping_rounds=20)
    train_predictions = model.predict_proba(x_train)[:, 1]
    test_predictions = model.predict_proba(x_test)[:, 1]
    fpr_insample, tpr_insample, _ = roc_curve(y_train, train_predictions)
    aucroc_insample = auc(fpr_insample, tpr_insample)
    fpr_outsample, tpr_outsample, _ = roc_curve(y_test, test_predictions)
    aucroc_outsample = auc(fpr_outsample, tpr_outsample)
    return aucroc_insample, aucroc_outsample


def main():
    xdf_train, ydf_train, ffeat_train, cfeat_train, targets = utils.load_data(
        'data/exp3_train.csv')
    xdf_val, ydf_val, ffeat_val, cfeat_val, targets = utils.load_data(
        'data/exp3_validation.csv')
    xdf_test, ydf_test, ffeat_test, cfeat_test, targets = utils.load_data(
        'data/exp3_test.csv')
    assert ffeat_train == ffeat_val
    assert ffeat_train == ffeat_test
    assert cfeat_train == cfeat_val
    assert cfeat_train == cfeat_test

    # ## Accuracy of prediction based on all features
    aucroc_insample, aucroc_outsample = train_and_evaluate(
        xdf_train, ydf_train,
        xdf_val, ydf_val,
        xdf_test, ydf_test,
        selection=None
    )
    print('\nAll features - No selection')
    print(f'Number of features: {xdf_train.shape[1]}')
    print(f'In-sample AUC-ROC score: {round(aucroc_insample, 4)}')
    print(f'Out-sample AUC-ROC score: {round(aucroc_outsample, 4)}')

    # ## Accuracy of prediction based on KSG selection
#     aucroc_insample_ksg, aucroc_outsample_ksg = train_and_evaluate(
#         xdf_train, ydf_train,
#         xdf_val, ydf_val,
#         xdf_test, ydf_test,
#         selection=ksg_selection,
#     )
#     print('\nKSG')
#     print(f'Number of features: {len(ksg_selection)}')
#     print(
#         f'In-sample AUC-ROC score with KSG selection: {round(aucroc_insample_ksg, 4)}')
#     print(
#         f'Out-sample AUC-ROC score with KSG selection: {round(aucroc_outsample_ksg, 4)}')

    # ## Accuracy of prediction based on HSIC selection
#     aucroc_insample_hsic, aucroc_outsample_hsic = train_and_evaluate(
#         xdf_train, ydf_train,
#         xdf_val, ydf_val,
#         xdf_test, ydf_test,
#         selection=hsic_selection,
#     )
#     print('\nHSIC Lasso')
#     print(f'Number of features: {len(hsic_selection)}')
#     print(
#         f'In-sample AUC-ROC score with HSIC Lasso selection: {round(aucroc_insample_hsic, 4)}')
#     print(
#         f'Out-sample AUC-ROC score with HSIC Lasso selection: {round(aucroc_outsample_hsic, 4)}')

    # ## Accuracy of prediction based on HISEL selection
#     aucroc_insample_hisel, aucroc_outsample_hisel = train_and_evaluate(
#         xdf_train, ydf_train,
#         xdf_val, ydf_val,
#         xdf_test, ydf_test,
#         selection=hisel_selection,
#     )
#     print('\nHISEL')
#     print(f'Number of features: {len(hisel_selection)}')
#     print(
#         f'In-sample AUC-ROC score with HISEL selection: {round(aucroc_insample_hisel, 4)}')
#     print(
#         f'Out-sample AUC-ROC score with HISEL selection: {round(aucroc_outsample_hisel, 4)}')

    # ## Accuracy of prediction based on boruta selection
#     aucroc_insample_boruta, aucroc_outsample_boruta = train_and_evaluate(
#         xdf_train, ydf_train,
#         xdf_val, ydf_val,
#         xdf_test, ydf_test,
#         selection=boruta_selection,
#     )
#     print('\nBORUTA')
#     print(f'Number of features: {len(boruta_selection)}')
#     print(
#         f'In-sample AUC-ROC score with Boruta selection: {round(aucroc_insample_boruta, 4)}')
#     print(
#         f'Out-sample AUC-ROC score with Boruta selection: {round(aucroc_outsample_boruta, 4)}')
#
    # ## Accuracy of prediction based on minerva selection
    aucroc_insample_minerva, aucroc_outsample_minerva = train_and_evaluate(
        xdf_train, ydf_train,
        xdf_val, ydf_val,
        xdf_test, ydf_test,
        selection=minerva_selection_1,
    )
    print('\nMINERVA')
    print(f'Number of features: {len(minerva_selection_1)}')
    print(
        f'In-sample AUC-ROC score with Minerva selection: {round(aucroc_insample_minerva, 4)}')
    print(
        f'Out-sample AUC-ROC score with Minerva selection: {round(aucroc_outsample_minerva, 4)}')
#


if __name__ == '__main__':
    main()

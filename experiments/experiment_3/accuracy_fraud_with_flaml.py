#!/usr/bin/env python
# coding: utf-8


from flaml import AutoML
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from sklearn.meatrics import auc, roc_curve  # NOQA

import utils
# from data.benchmark_selection import (
#     ksg_selection,
#     hsic_selection,
#     boruta_selection,
#     hisel_selection,
# )
from data.minerva_selection import (
    minerva_selection_2,
)


automl_fit_parameters = dict(
    time_budget=20 * 60,
    metric='ap',
    early_stop=True,
)


def log_positive_frequency(ydf):
    freq = sum(ydf['y'] == 1) / len(ydf)
    print(f'Label distribution:\n{ydf.describe()}')
    print(f'Frequency of positive labels: {round(freq, 6)}')


def log_positive_frequencies(ydf_train, ydf_val, ydf_test):
    print('\nTRAIN')
    log_positive_frequency(ydf_train)
    print('\nVALIDATION')
    log_positive_frequency(ydf_val)
    print('\nTEST')
    log_positive_frequency(ydf_test)


def load_chron_data(test_size=.2, val_size=.2):
    xdf, ydf, ffeat, cfeat, targets = utils.load_data(
        'data/exp3_chron.csv')
    n_samples = len(xdf)
    train_idx = int((1. - test_size - val_size) * n_samples)
    val_idx = train_idx + int(n_samples * val_size)
    xdf_train = xdf.iloc[:train_idx].copy()
    ydf_train = ydf.iloc[:train_idx].copy()
    xdf_val = xdf.iloc[train_idx:val_idx].copy()
    ydf_val = ydf.iloc[train_idx:val_idx].copy()
    xdf_test = xdf.iloc[val_idx:].copy()
    ydf_test = ydf.iloc[val_idx:].copy()
    return (
        xdf_train,
        ydf_train,
        xdf_val,
        ydf_val,
        xdf_test,
        ydf_test,
    )


def load_shuffled_data():
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
    return (
        xdf_train,
        ydf_train,
        xdf_val,
        ydf_val,
        xdf_test,
        ydf_test,
    )


def train_and_evaluate(X_train, y_train, X_test, y_test, selection=None):
    if selection is None:
        selection = X_train.columns.tolist()
    x_train = X_train.loc[:, selection].copy()
    x_test = X_test.loc[:, selection].copy()
    model = AutoML()
    model.fit(x_train, y_train['y'],
              task='classification', **automl_fit_parameters)
    train_predictions = model.predict(x_train)
    train_prediction_probabilities = model.predict_proba(x_train)[:, 1]
    test_predictions = model.predict(x_test)
    test_prediction_probabilities = model.predict_proba(x_test)[:, 1]

    recall_insample = recall_score(y_train, train_predictions)
    precision_insample = precision_score(y_train, train_predictions)
    inpre, inrec, _ = precision_recall_curve(
        y_train, train_prediction_probabilities)
    aucprc_insample = auc(inrec, inpre)

    recall_outsample = recall_score(y_test, test_predictions)
    precision_outsample = precision_score(y_test, test_predictions)
    outpre, outrec, _ = precision_recall_curve(
        y_test, test_prediction_probabilities)
    aucprc_outsample = auc(outrec, outpre)

#     fpr_insample, tpr_insample, _ = roc_curve(y_train, train_prediction_probabilities)
#     aucroc_insample = auc(fpr_insample, tpr_insample)

#     fpr_outsample, tpr_outsample, _ = roc_curve(y_test, test_prediction_probabilities)
#     aucroc_outsample = auc(fpr_outsample, tpr_outsample)
    return recall_insample, precision_insample, recall_outsample, precision_outsample, aucprc_insample, aucprc_outsample, model


def main():
    (xdf_train,
        ydf_train,
        _,
        ydf_val,
        xdf_test,
        ydf_test,
     ) = load_chron_data()
    log_positive_frequencies(ydf_train, ydf_val, ydf_test)

    # ## Accuracy of prediction based on all features
    recall_insample, precision_insample, recall_outsample, precision_outsample, aucprc_insample, aucprc_outsample, model = train_and_evaluate(
        xdf_train, ydf_train,
        xdf_test, ydf_test,
        selection=None
    )
    print('\nAll features - No selection')
    print(f'Number of features: {xdf_train.shape[1]}')
    print(f'In-sample recall: {round(recall_insample, 4)}')
    print(f'Out-sample recall: {round(recall_outsample, 4)}')
    print(f'In-sample precision: {round(precision_insample, 4)}')
    print(f'Out-sample precision: {round(precision_outsample, 4)}')
    print(f'In-sample AUC-PRC score: {round(aucprc_insample, 4)}')
    print(f'Out-sample AUC-PRC score: {round(aucprc_outsample, 4)}')

    # ## Accuracy of prediction based on minerva selection
    recall_insample_minerva, precision_insample_minerva, recall_outsample_minerva, precision_outsample_minerva, aucprc_insample_minerva, aucprc_outsample_minerva, _ = train_and_evaluate(
        xdf_train, ydf_train,
        xdf_val, ydf_val,
        xdf_test, ydf_test,
        selection=minerva_selection_2,
    )
    print('\nMINERVA')
    print(f'Number of features: {len(minerva_selection_2)}')
    print(
        f'In-sample recall with Minerva selection: {round(recall_insample_minerva, 4)}')
    print(
        f'Out-sample recall with Minerva selection: {round(recall_outsample_minerva, 4)}')
    print(
        f'In-sample precision with Minerva selection: {round(precision_insample_minerva, 4)}')
    print(
        f'Out-sample precision with Minerva selection: {round(precision_outsample_minerva, 4)}')
    print(
        f'In-sample AUC-PRC score with Minerva selection: {round(aucprc_insample_minerva, 4)}')
    print(
        f'Out-sample AUC-PRC score with Minerva selection: {round(aucprc_outsample_minerva, 4)}')
    return model


# In[ ]:

if __name__ == '__main__':
    _ = main()


# In[ ]:

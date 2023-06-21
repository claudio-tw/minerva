from typing import List, Optional, Union
import re
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import pyHSICLasso


class FeatureType(Enum):
    CATEGORICAL = 0
    FLOAT = 1


def _preprocess_datatypes(
        y: Union[pd.DataFrame, pd.Series],
) -> Union[pd.DataFrame, pd.Series]:
    if isinstance(y, pd.DataFrame):
        for col in y.columns:
            if y[col].dtype == bool:
                y[col] = y[col].astype(int)
    elif y.dtypes == bool:
        y = y.astype(int)
    ydtypes = y.dtypes if isinstance(y, pd.DataFrame) else [y.dtypes]
    for dtype in ydtypes:
        assert dtype == int or dtype == float
    return y


def ksgmi(
        x: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        threshold: float = .01,
):
    x = _preprocess_datatypes(x)
    y = _preprocess_datatypes(y)
    discrete_features = x.dtypes == int
    mix = x.values
    if isinstance(y, pd.Series) or (isinstance(y, pd.DataFrame) and y.shape[1] == 1):
        miy = np.squeeze(y.values)
    else:
        miy = np.sum(y, axis=1)  # reduce to one-dimensional target
    compute_mi = mutual_info_classif if miy.dtype == int else mutual_info_regression
    mis = compute_mi(mix, miy, discrete_features=discrete_features)
    max_mi = float(np.max(mis))
    normalization = max_mi if not np.isclose(max_mi, 0) else 1.
    mis /= normalization
    isrelevant = mis > threshold
    relevant_features = np.arange(x.shape[1])[isrelevant]
    print(
        f'ksg-mi preprocessing: {sum(isrelevant)} features have been selected')
    return relevant_features, mis


def pyhsiclasso(x, y, xfeattype,  yfeattype, n_features: int, batch_size=500):
    lasso = pyHSICLasso.HSICLasso()
    lasso.X_in = x.T
    lasso.Y_in = y.T
    discrete_x = xfeattype == FeatureType.CATEGORICAL
    if yfeattype == FeatureType.CATEGORICAL:
        lasso.classification(n_features, B=batch_size, discrete_x=discrete_x)
    else:
        lasso.regression(n_features, B=batch_size, discrete_x=discrete_x)
    return lasso.A


def onedimlabel(x):
    assert x.ndim == 2
    ns = np.amax(x, axis=0)
    res = np.array(x[:, 0], copy=True)
    m = 1
    for i in range(1, x.shape[1]):
        m *= max(1, ns[i-1])
        res += (1+m) * x[:, i]
    return res


def load_data(path: Union[str, Path] = 'data/featsel_cache.csv'):
    df = pd.read_csv(path)
    # 1. Drop columns whose entries are all NaN
    targets = [
        'VOLUME3M_TARGET',
        'TRANSFER3M_TARGET',
        'LTV3M_TARGET'
    ]
    target = 'TRANSFER3M_TARGET'
    to_drop = [
        'PROFILE_ID',
        'TEST_GROUP',
        'ASSIGNMENT_DT',
        'USER_ID'
    ]
    to_drop += [col for col in df.columns if (
        col not in targets) and (
            (re.search('[3456789]M', col) is not None) or
            (re.search('[12][0-9]M', col) is not None)
    )
    ]
    df = df.drop(to_drop, axis=1)
    cnt = df.isna().sum()
    idxallna = cnt > .95 * len(df)
    allnacols = list(cnt.loc[idxallna].index)
    df.drop(columns=allnacols, inplace=True)
    print(f'Dimensionality of dataset after dropping: {df.shape}')
    # 2. Convert count et similar features to float so that we use as continuous features
    tofloat = [
        'COUNT_BALANCES',
        'COUNT_BALANCE_CURRENCIES',
        'COUNT_BALANCE_TYPES',
        'COUNT_CCY_ACTIONS',
        'COUNT_DISTINCT_CCY_SM_ROUTES',
        'COUNT_DISTINCT_XCCY_SM_ROUTES',
        'COUNT_UNSUCCESSFUL_ACTIONS',
        'COUNT_XCCY_ACTIONS',
        'DAYS_SINCE_LAST_BALANCE',
        'DAYS_SINCE_LAST_CARD',
        'DAYS_SINCE_LAST_SENDMONEY',
    ]
    df[tofloat] = df[tofloat].astype(float)
    # 3. Ordinal encoding of categorical features
    cattypes = ['object']
    catcols = list(df.select_dtypes(include=cattypes).columns)
    ordinal_encoder = ce.OrdinalEncoder(cols=catcols)
    ordinal_encoder.fit(df)
    df = ordinal_encoder.transform(df)
    idxfloat = (df.dtypes == float)
    floatcols = list(df.dtypes.loc[idxfloat].index)
    intcols = list(set(df.columns).difference(set(floatcols)))
    df[floatcols] = df[floatcols].fillna(.0).astype(float)
    df[intcols] = df[intcols].fillna(0).astype(int)
    float_features = floatcols
    cat_features = intcols
    all_targets = [
        'VOLUME3M_TARGET',
        'TRANSFER3M_TARGET',
        'LTV3M_TARGET'
    ]
    for target in all_targets:
        float_features.remove(target)
    targets = [
        'TRANSFER3M_TARGET',
    ]
    ydf = df.loc[:, targets]
    xdf = df.loc[:, float_features + cat_features]
    return xdf, ydf, float_features, cat_features

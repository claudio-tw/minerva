from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


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
        miy = np.linalg.norm(y, axis=1)
    compute_mi = mutual_info_classif if miy.dtype == int else mutual_info_regression
    mis = compute_mi(mix, miy, discrete_features=discrete_features)
    mis /= np.max(mis)
    isrelevant = mis > threshold
    relevant_features = np.arange(x.shape[1])[isrelevant]
    print(f'ksg-mi preprocessing: {sum(isrelevant)} features are pre-selected')
    return relevant_features, mis

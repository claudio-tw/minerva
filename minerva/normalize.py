from typing import Sequence, Optional, List, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import OrdinalEncoder


def ecdf(x: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys


class Transformer:
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class ECDFNormalizer1D(Transformer):
    def __init__(self):
        pass

    def fit(self, x: Sequence[float]):
        xs, ys = ecdf(np.array(x))
        self.interp = interp1d(xs, ys, bounds_error=False, fill_value=(0., 1.))

    def transform(self, x: Sequence[float]):
        return self.interp(np.array(x))


class ECDFNormalizer(Transformer):
    def __init__(self, cols: Optional[List[str]] = None):
        self.norms = defaultdict(ECDFNormalizer1D)
        self.cols = cols

    def fit(self, X: pd.DataFrame):
        if self.cols is None:
            self.cols = X.select_dtypes(include=[np.float])

        for c in self.cols:
            self.norms[c].fit(X[c].values)

    def transform(self, X: pd.DataFrame):
        out = X.copy()
        for c in self.norms.keys():
            out[c] = self.norms[c].transform(X[c].values)
        return out


class DatasetNormalizer(Transformer):
    """
    ECDF-normalizes floating-point features, ordinal-encodes categorical ones
    """

    def __init__(self, float_cols: List[str], categorical_cols: List[str]):
        self.float_enc = ECDFNormalizer(float_cols)
        self.cat_enc = OrdinalEncoder(encoded_missing_value=-1)
        self.cat_cols = categorical_cols

    def fit(self, X: pd.DataFrame):
        self.float_enc.fit(X)
        self.cat_enc.fit(X[self.cat_cols])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.float_enc.cols] = self.float_enc.transform(
            X[self.float_enc.cols])
        X[self.cat_cols] = self.cat_enc.transform(X[self.cat_cols])
        return X

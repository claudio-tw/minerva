from typing import Union
from pathlib import Path
import pandas as pd


def load_data(path: Union[Path, str] = 'data/exp2large.csv'):
    df = pd.read_csv(path)
    float_features = [col for col in df if col.startswith('xf')]
    cat_features = [col for col in df if col.startswith('xc')]
    targets = ['y']
    xdf = df[cat_features + float_features]
    ydf = df[targets].astype(float)
    return xdf, ydf, float_features, cat_features, targets

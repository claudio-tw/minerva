from typing import Union
from pathlib import Path
import pandas as pd


def load_data(path: Union[Path, str] = 'data/exp2.csv'):
    df = pd.read_csv(path)
    float_features = [col for col in df if col.startswith('xf')]
    cat_features = [col for col in df if col.startswith('xc')]
    targets = ['y']
    xdf = df[cat_features + float_features]
    ydf = df[targets]
    return xdf, ydf, float_features, cat_features, targets

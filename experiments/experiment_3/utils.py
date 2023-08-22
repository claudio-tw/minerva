from typing import Union
from pathlib import Path
import datetime
import pandas as pd
import numpy as np


def load_data(path: Union[Path, str] = 'data/exp3.csv'):
    if isinstance(path, str):
        path = Path(path)
    lastmod_ = path.stat().st_mtime
    lastmodified = datetime.datetime.fromtimestamp(lastmod_)
    print(f'Loading {path}... Last modified date: {lastmodified}')
    df = pd.read_csv(path)
    float_features = [col for col in df if col.startswith('xf')]
    cat_features = [col for col in df if col.startswith('xc')]
    targets = ['y']
    xdf = df[cat_features + float_features].copy()
    xdf[cat_features] = xdf[cat_features].astype(int)
    xdf[float_features] = xdf[float_features].astype(float)
    ydf = df[targets].astype(int)
    print(f'Number of categorical features: {len(cat_features)}')
    print(f'Number of float features: {len(float_features)}')
    print(f'Number of samples: {len(df)}')
    ys = np.array(ydf['y'].values, dtype=int)
    ys_ = 1 - ys
    for f in cat_features + float_features:
        if np.allclose(ys, xdf[f], rtol=1e-2, atol=1e-2):
            print(f'It seems that feature {f} is similar to the target!')
        if np.allclose(ys_, xdf[f], rtol=1e-2, atol=1e-2):
            print(
                f'It seems that feature {f} is similar to the opposite of the target!')
    return xdf, ydf, float_features, cat_features, targets

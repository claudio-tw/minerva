import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import adjusted_mutual_info_score

import tools


def main():
    # Load dataframe
    d = 30
    df = pd.read_csv('data/large.csv')
    n_samples = len(df)
    expected_features = np.array([3, 8])
    features = [f'f{n}' for n in range(d)]
    targets = ['y']
    xdf = df[features]
    ydf = df[targets]
    x = xdf.values
    y = ydf.values

    # Uncover the dependence between target and features
    test = np.array(x[:, expected_features[0]] ==
                    x[:, expected_features[1]], dtype=int)
    assert np.all(test == y[:, 0])

    # Preliminary check: expected features bear the highest information content
    l = 2
    miscores = {subset:
                adjusted_mutual_info_score(
                    tools.onedimlabel(x[:, list(subset)]), y[:, 0])
                for subset in itertools.combinations(list(range(d)), l)

                }
    s = (0, 1)
    mi = 0
    for k, v in miscores.items():
        if v > mi:
            s = k
            mi = v
    highest_info = s
    print(f'Expected features: {sorted(expected_features)}')
    print(
        f'Pair of features with highest information content: {sorted(highest_info)}')

    # Selection with marginal 1D ksg mutual info
    ksgselection, mis = tools.ksgmi(xdf, ydf, threshold=0.05)
    print(f'Expected features: {sorted(expected_features)}')
    print(f'Marginal KSG selection: {sorted(ksgselection)}')

    # Selection with HSIC Lasso
    xfeattype = tools.FeatureType.CATEGORICAL
    yfeattype = tools.FeatureType.CATEGORICAL
    hsiclasso_selection = tools.pyhsiclasso(
        x, y, xfeattype=xfeattype, yfeattype=yfeattype, n_features=2, batch_size=500)
    print(f'Expected features: {sorted(expected_features)}')
    print(f'HSIC Lasso selection: {sorted(hsiclasso_selection)}')


if __name__ == '__main__':
    main()

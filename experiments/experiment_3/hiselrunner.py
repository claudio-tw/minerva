#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import pickle
import pandas as pd
import numpy as np

import hisel
import utils


def main(data_path='data/exp3.csv'):
    xdf, ydf, float_features, cat_features, targets = utils.load_data(
        data_path)
    search_parameters = hisel.feature_selection.SearchParameters(
        num_permutations=10,
        im_ratio=.01,
        max_iter=2,
        parallel=True,
        random_state=None,
    )
    hsiclasso_parameters = hisel.feature_selection.HSICLassoParameters(
        mi_threshold=.001,
        hsic_threshold=.01,
        batch_size=5000,
        minibatch_size=500,
        number_of_epochs=2,
        use_preselection=True,
        device='cpu',
    )
    res = hisel.feature_selection.select_features(
        xdf,
        ydf,
        hsiclasso_parameters,
        search_parameters
    )
    print(f"HISEL selection:\n{res['selected_features']}")


if __name__ == '__main__':
    main()

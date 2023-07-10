#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from arfs.feature_selection import allrelevant
from arfs.feature_selection.allrelevant import Leshy
from xgboost import XGBRegressor

import hisel

import tools
from experiment_2 import utils


def main():
    xdf, ydf, float_features, cat_features, targets = utils.load_data(
        'data/exp2filtered.csv')
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
        batch_size=9000,
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
    return res

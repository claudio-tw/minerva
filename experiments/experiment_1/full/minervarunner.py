import numpy as np
import pandas as pd
import pickle
import torch

import minerva
from minerva.select import Selector


def main():
    n = 100000
    dy = 1
    num_cat_features = 10
    num_cont_features = 30
    feature_cols = [f'x{n}' for n in range(
        num_cat_features + num_cont_features)]
    cat_features = feature_cols[:num_cat_features]
    float_features = feature_cols[num_cat_features:]
    targets = [f'y{n}' for n in range(dy)]

    data = pd.read_csv('data/large.csv')
    xdf = data.loc[:, feature_cols]
    x = xdf.values
    ydf = data.loc[:, targets]
    y = ydf.values
    cat_feat_sizes = 1+data.loc[:, cat_features].max().values
    store = pickle.load(open('data/store.exp2', 'rb'))

    expected_cat = store['expected_cat']
    expected_cont0 = store['expected_cont0']
    expected_cont1 = store['expected_cont1']
    expected_cont = store['expected_cont']
    expected_features = store['expected_features']

    # ### Uncover relation between features and data
    _chooser = data.iloc[:, expected_cat[1]] == data.iloc[:, expected_cat[0]]
    idx0 = _chooser == 0
    idx1 = _chooser == 1
    y_ = np.zeros(shape=(len(data), dy))
    y_[idx0, :] = (
        store['t0'] @ np.expand_dims(
            np.sin(2 * np.pi * data.loc[idx0].iloc[:, expected_cont0]),
            axis=2))[:, :, 0]
    y_[idx1, :] = (
        store['t1'] @ np.expand_dims(
            np.cos(2 * np.pi * data.loc[idx1].iloc[:, expected_cont1]),
            axis=2))[:, :, 0]
    assert np.allclose(np.squeeze(y_), data['y0'].values, atol=1e-6, rtol=1e-4)

    # Split train, validation, and test
    n_samples = len(data)
    train_size = int(.70 * n_samples)
    val_size = int(.25 * n_samples)
    test_size = n_samples - train_size - val_size
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size: train_size + val_size]
    test_data = data.iloc[:-test_size]

    # Set hyperparameters
    projection_init = np.array(
        [.90] * num_cat_features + [.95] * num_cont_features
    )

    # Design architecture
    dimension_of_residual_block = 512
    num_res_layers = 4
    scaler = 2
    emb_dim = 4
    reg_coef = 1e0

    # Batches and epochs
    max_epochs = int(2000*scaler)
    batch_size = scaler*1200

    # Pack hyperparameters
    selector_params = dict(
        cat_features=cat_features,
        float_features=float_features,
        targets=targets,
        dim1_max=dimension_of_residual_block,
        num_res_layers=num_res_layers,
        eps=.001,
        cat_feat_sizes=cat_feat_sizes,
        emb_dim=emb_dim,
    )
    logger_params = dict(
        name="experiment_1_full"
    )

    # No-regularisation train control
    noreg_train_control = minerva.feature_selection.TrainControl(
        model_name='exp1full_noreg',
        data_path='data/',
        number_of_epochs=max_epochs,
        number_of_segments=5,
        learning_rate=5e-6,
        reg_coef=.0,
        projection_init=projection_init,
        disable_projection=True,
    )

    # Selection train control
    select_train_control = minerva.feature_selection.TrainControl(
        model_name='exp1full_sel',
        data_path='data/',
        number_of_epochs=max_epochs,
        number_of_segments=3,
        learning_rate=1e-7,
        reg_coef=1e0,
        projection_init=None,
        disable_projection=False,
    )

    selector = minerva.feature_selection.run(
        train_data,
        val_data,
        test_data,
        float_features,
        cat_features,
        targets,
        selector_params,
        logger_params,
        noreg_train_control,
        select_train_control,
        batch_size,
    )

    print(f'Expected features: {expected_features}')
    print(f'Selected features: {selector.selected_feature_names()}')


if __name__ == '__main__':
    main()

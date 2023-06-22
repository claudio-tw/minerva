import numpy as np
import pandas as pd
import torch

import minerva


def main():
    # Load dataframe
    d = 30
    df = pd.read_csv('data/large.csv')
    n_samples = len(df)
    expected_features = np.array([3, 8])

    feature_cols = [f'f{n}' for n in range(d)]
    cat_features = feature_cols  # all features are categorical
    float_features = []  # no feature is float
    targets = ['y']
    cat_feat_sizes = 1 + df[cat_features].max().values
    train_size = int(.75 * n_samples)
    val_size = int(.225 * n_samples)
    test_size = n_samples - train_size - val_size

    # Split train, validation, and test
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size: train_size + val_size]
    test_data = df.iloc[:-test_size]

    # Design architecture
    dimension_of_residual_block = 512
    num_res_layers = 4
    scaler = 2
    emb_dim = 4

    # Batches and epochs
    max_epochs = int(2500*scaler)
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
        name="experiment_1_cat_only"
    )

    # No-regularisation train control
    noreg_train_control = minerva.feature_selection.TrainControl(
        model_name='expco_noreg_pe',
        data_path='data/',
        number_of_epochs=2000,
        number_of_segments=1,
        learning_rate=5e-6,
        reg_coef=.0,
        projection_init=.20,
        disable_projection=False,
        first_run_load_path='data/expco_noreg.0',
    )

    # Selection train control
    select_train_control = minerva.feature_selection.TrainControl(
        model_name='expco_sel',
        data_path='data/',
        number_of_epochs=4000,
        number_of_segments=2,
        learning_rate=5e-6,
        reg_coef=1e6,
        projection_init=None,
        disable_projection=False,
    )

    _ = minerva.feature_selection.run(
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


if __name__ == '__main__':
    main()

import os
import sys
import numpy as np
import pandas as pd

import minerva
import utils


def main():
    xdf, ydf, float_features, cat_features, targets = utils.load_data(
        'data/exp3_chron.csv')
    num_cat_features = len(cat_features)
    num_cont_features = len(float_features)
    data = pd.concat((xdf, ydf), axis=1)
    cat_feat_sizes = 1+data.loc[:, cat_features].max().values

    # Split train, validation, and test
    n_samples = len(data)
    train_size = int(.70 * n_samples)
    val_size = int(.20 * n_samples)
    test_size = n_samples - train_size - val_size
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size: train_size + val_size]
    test_data = data.iloc[:-test_size]

    # Set hyperparameters
    projection_init = np.array(
        [.1] * num_cat_features + [.1] * num_cont_features
    )

    # Design architecture
    dimension_of_residual_block = 512
    num_res_layers = 4
    scaler = 2
    emb_dim = 4

    # Batches and epochs
    max_epochs = int(2000*scaler)
    batch_size = scaler*800

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
        name="cardfraud"
    )
    # No-regularisation train control
    noreg_train_control = minerva.feature_selection.TrainControl(
        model_name='cardfraud_noreg',
        data_path='data/',
        number_of_epochs=500,
        number_of_segments=1,
        learning_rate=1e-6,
        reg_coef=.0,
        projection_init=projection_init,
        disable_projection=True,
        first_run_load_path=None,
    )

    # Selection train control
    select_train_control = minerva.feature_selection.TrainControl(
        model_name='cardfraud_sel',
        data_path='data/',
        number_of_epochs=100,
        number_of_segments=8,
        learning_rate=1e-6,
        reg_coef=1e3,
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

    print(f'Selected features: {selector.selected_feature_names()}')
    pickle.dump(selector.selected_feature_names(),
                open('data/minerva_selection', 'wb'))
    console_stdout = sys.stdout
    sys.stdout = open('data/minerva_selection.txt', 'w')
    for feature in selector.selected_feature_names():
        sys.stdout.write(feature + '\n')
    sys.stdout.close()
    sys.stdout = console_stdout


if __name__ == '__main__':
    main()

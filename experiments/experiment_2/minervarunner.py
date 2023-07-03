import numpy as np
import pandas as pd

import minerva
import tools


def main():
    xdf, ydf, float_features, cat_features = tools.load_transfer_data(
        'data/transfer3m.csv')
    targets = ['TRANSFER3M_TARGET']
    num_cat_features = len(cat_features)
    num_cont_features = len(float_features)
    data = pd.concat((xdf, ydf), axis=1)
    cat_feat_sizes = 1+data.loc[:, cat_features].max().values

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
        [.10] * num_cat_features + [.10] * num_cont_features
    )

    # Design architecture
    dimension_of_residual_block = 512
    num_res_layers = 4
    scaler = 2
    emb_dim = 4

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
        name="transfer3m"
    )

    # No-regularisation train control
    noreg_train_control = minerva.feature_selection.TrainControl(
        model_name='transfer3m_noreg',
        data_path='data/',
        number_of_epochs=4500,
        number_of_segments=1,
        learning_rate=5e-6,
        reg_coef=.0,
        projection_init=projection_init,
        disable_projection=True,
        first_run_load_path=None,
    )

    # Selection train control
    select_train_control = minerva.feature_selection.TrainControl(
        model_name='transfer3m_sel',
        data_path='data/',
        number_of_epochs=max_epochs,
        number_of_segments=5,
        learning_rate=1e-7,
        reg_coef=1e5,
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


if __name__ == '__main__':
    main()

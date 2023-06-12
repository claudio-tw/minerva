import numpy as np
import pandas as pd
import pickle
import torch

import minerva


def main():
    n = 50000
    dy = 1
    num_cat_features = 10
    num_cont_features = 30
    feature_cols = [f'x{n}' for n in range(
        num_cat_features + num_cont_features)]
    cat_features = feature_cols[:num_cat_features]
    cat_feat_sizes = 1+data.loc[:, cat_features].max().values
    float_features = feature_cols[num_cat_features:]
    targets = [f'y{n}' for n in range(dy)]

    data = pd.read_csv('data/large.csv')
    xdf = data.loc[:, feature_cols]
    x = xdf.values
    ydf = data.loc[:, targets]
    y = ydf.values
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
    train_size = int(.75 * n_samples)
    val_size = int(.225 * n_samples)
    test_size = n_samples - train_size - val_size
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size: train_size + val_size]
    test_data = data.iloc[:-test_size]

    # Set hyperparameters
    dimension_of_residual_block = 512
    num_res_layers = 4
    scaler = 2
    batch_size = scaler*1200
    num_batches = n_samples // batch_size
    max_epochs = int(2000*scaler)
    lr = 5e-6
    emb_dim = 4
    reg_coef = 1e6

    # Pack hyperparameters
    selector_params = dict(
        cat_features=cat_features,
        float_features=float_features,
        targets=targets,
        dim1_max=dimension_of_residual_block,
        lr=lr,
        num_res_layers=num_res_layers,
        eps=.001,
        cat_feat_sizes=cat_feat_sizes,
        emb_dim=emb_dim,
    )
    logger_params = dict(
        name="experiment_2"
    )

    # Set dataloaders
    train_dataloader, val_dataloader, test_dataloader = minerva.feature_selection.dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        float_features=float_features,
        categorical_features=cat_features,
        targets=targets,
        batch_size=batch_size,
    )

    logs = []
    # First pass: No regularisation
    noreg_path = 'data/noreg.model'
    out, selector = minerva.feature_selection.train(
        selector_params=selector_params,
        logger_params=logger_params,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        reg_coef=.0,
        projection_init=.20,
        disable_projection=False,
        max_epochs=max_epochs,
        load_path=None
    )
    logs.append(out)
    torch.save(selector.state_dict(), noreg_path)
    previous_segment_path = noreg_path

    # Second pass: Apply regularisation
    for segment in range(5):
        out, selector = minerva.feature_selection.train(
            selector_params=selector_params,
            logger_params=logger_params,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            reg_coef=reg_coef,
            disable_projection=False,
            max_epochs=max_epochs,
            load_path=previous_segment_path
        )
        segment_path = f'data/trained.model.{segment}.0'
        torch.save(selector.state_dict(), segment_path)
        logs.append(out)
        previous_segment_path = segment_path

    dflogs = pd.DataFrame(logs)
    dflogs.to_csv('data/traininglogs.csv', index=False)


if __name__ == '__main__':
    main()

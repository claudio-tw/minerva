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
    dimension_of_residual_block = 512
    num_res_layers = 4
    scaler = 2
    batch_size = scaler*1200
    num_batches = n_samples // batch_size
    max_epochs = int(2000*scaler)
    emb_dim = 4
    reg_coef = 1e0

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
    noreg_selector_params = selector_params.copy()
    noreg_selector_params['lr'] = 5e-6
    noreg_selector_params['mi_threshold'] = None
    reg_selector_params = selector_params.copy()
    reg_selector_params['lr'] = 1e-7
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
    noreg_path = 'data/run3/noreg.model.6.4'
    for segment in range(1, 2):
        load_path = None if segment == 0 else noreg_path
        out, selector = minerva.feature_selection.train(
            selector_params=noreg_selector_params,
            logger_params=logger_params,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            reg_coef=.0,
            projection_init=projection_init,
            disable_projection=False,
            max_epochs=800,
            load_path=load_path
        )
        logs.append(out)
        noreg_path = f'data/noreg.model.8.{segment}'
        print(f'Saving state dict to {noreg_path}')
        torch.save(selector.state_dict(), noreg_path)

    previous_segment_path = noreg_path
    # Second pass: Apply regularisation
    for segment in range(4):
        out, selector = minerva.feature_selection.train(
            selector_params=selector_params,
            logger_params=logger_params,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            reg_coef=reg_coef,
            projection_init=None,
            disable_projection=False,
            max_epochs=800,
            load_path=previous_segment_path
        )
        segment_path = f'data/trained.model.8.{segment}.0'
        print(f'Saving state dict to {segment_path}')
        torch.save(selector.state_dict(), segment_path)
        logs.append(out)
        dflogs = pd.DataFrame(logs)
        dflogs.to_csv('data/traininglogs8.csv', index=False)
        weights = selector.projection_weights()
        weight_history = pd.DataFrame(selector.weight_history)
        weight_history.to_csv(
            f'data/weight_history_8_segment{segment}.csv', index=False)
        print(
            f'train_mutual_information: {float(selector.train_mutual_information())}')
        print(
            f'val_mutual_information: {float(selector.val_mutual_information())}')
        print(f'weights:\n{weights}\n')
        print(f'expected_cat:\n{expected_cat}\n')
        print(f'expected_cont0:\n{expected_cont0}\n')
        print(f'expected_cont1:\n{expected_cont1}\n')
        print(f'expected_features:\n{expected_features}\n')
        print(f'selected features:\n{selector.selected_feature_names()}\n')
        print(f'logs:\n{dflogs}\n')

        previous_segment_path = segment_path


if __name__ == '__main__':
    main()

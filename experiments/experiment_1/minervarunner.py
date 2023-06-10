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
        name="experiment_1"
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

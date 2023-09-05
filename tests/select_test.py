import unittest
from typing import Sequence
import numpy as np
import pandas as pd
from minerva import feature_selection
from minerva.select import Selector


class SelectTest(unittest.TestCase):

    @staticmethod
    def synthesize_categorical_data(
        n: int,
        dx: int,
        num_relevant: int,
        feat_sizes: Sequence[int],
        train_size: int,
        val_size: int,
        test_size: int,
    ):
        xs = [
            np.random.randint(low=0, high=size, size=(n, 1))
            for size in feat_sizes
        ]
        x = np.concatenate(xs, axis=1)
        expected = np.random.choice(dx, replace=False, size=num_relevant)
        y = np.zeros(shape=(n,), dtype=int)
        for f0, f1 in zip(expected[:-1], expected[1:]):
            x0 = x[:, f0] / feat_sizes[f0]
            x1 = x[:, f1] / feat_sizes[f1]
            y += np.array(x0 > x1, dtype=int)

        feature_cols = [f'f{n}' for n in range(dx)]
        targets = ['y']
        xdf = pd.DataFrame(
            x,
            columns=feature_cols
        )
        ydf = pd.DataFrame(
            y,
            columns=targets
        )
        data = pd.concat((xdf, ydf), axis=1)
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size: train_size + val_size]
        test_data = data.iloc[train_size + val_size:]

        return expected, train_data, val_data, test_data

    def test_mutual_info_categorical_features(self):
        n = 10000
        dx = 20
        num_relevant = 4
        feat_sizes = [10]*dx
        train_size = int(.60 * n)
        val_size = int(.20 * n)
        test_size = int(.20 * n)
        feature_cols = [f'f{n}' for n in range(dx)]
        cat_features = feature_cols  # all features are categorical
        float_features = []  # No feature is float
        targets = ['y']

        # Metaparameters
        dimension_of_residual_block = 512
        num_res_layers = 4
        emb_dim = 3
        batch_size = 750

        # Pack hyperparameters
        selector_params = dict(
            cat_features=cat_features,
            float_features=float_features,
            targets=targets,
            dim1_max=dimension_of_residual_block,
            num_res_layers=num_res_layers,
            cat_feat_sizes=feat_sizes,
            emb_dim=emb_dim,
        )

        expected, train_data, val_data, test_data = SelectTest.synthesize_categorical_data(
            n, dx, num_relevant, feat_sizes,  train_size, val_size, test_size)

        train_dataloader, val_dataloader, test_dataloader = feature_selection.dataloaders(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            float_features=float_features,
            categorical_features=cat_features,
            targets=targets,
            batch_size=batch_size,
        )

        selector = Selector(**selector_params)

        # Set dataloaders
        selector.set_loaders(train_dataloader, val_dataloader, test_dataloader)
        selector.disable_projection()

        # Compute
        train_mi = float(selector.train_mutual_information())
        val_mi = float(selector.val_mutual_information())
        test_mi = float(selector.test_mutual_information())
        self.assertTrue(isinstance(train_mi, float))
        self.assertTrue(isinstance(val_mi, float))
        self.assertTrue(isinstance(test_mi, float))


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import pandas as pd
from scipy.stats import norm

from minerva import normalize
from minerva.iterable_dataset import MyDataset


class NormalizeTest(unittest.TestCase):
    def test_ecdnormalizer1d(self):
        loc = np.random.uniform(low=-100, high=100)
        scale = np.random.uniform(low=.1, high=10.)
        n_samples = 100000
        sampler = norm(loc=loc, scale=scale)
        samples = sampler.rvs(size=n_samples)
        normalizer = normalize.ECDFNormalizer1D()
        normalizer.fit(samples)
        normalized_samples = normalizer.transform(samples)
        exact_cdf = sampler.cdf(samples)
        adiff = np.abs(exact_cdf - normalized_samples)
        rdiff = adiff / np.abs(exact_cdf)
        msg = 'Exact cdf and empirical cdf do not reconcile with each other.\n'
        msg += f'Maximum absolute difference: {adiff.max()}\n'
        msg += f'Maximum relative difference: {rdiff.max()}\n'
        self.assertTrue(
            np.allclose(
                normalized_samples,
                exact_cdf,
                rtol=5e-2,
                atol=1e-3,
            ),
            msg
        )

    def test_all_categorical(self):
        # Parameters for the synthetis of data
        n = 50000
        dx = 10
        num_relevant = 2
        feat_sizes = np.random.randint(low=2, high=15, size=(dx))
        dy = 1

        # Synthesize the data
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
            y ^= np.array(x0 > x1, dtype=bool)

        feature_cols = [f'f{n}' for n in range(dx)]
        float_features = []
        cat_features = feature_cols
        targets = [f'y{n}' for n in range(dy)]
        targets = targets
        xdf = pd.DataFrame(
            x,
            columns=feature_cols
        )
        ydf = pd.DataFrame(
            y,
            columns=targets
        )
        df = pd.concat((xdf, ydf), axis=1)
        # Prepare the data for the training
        dn = normalize.DatasetNormalizer(
            float_cols=[], categorical_cols=cat_features + targets)
        data = dn.fit_transform(df)

        dataset = MyDataset(
            data,
            float_features,
            cat_features,
            targets
        )

        self.assertTrue(
            np.allclose(
                np.array(df.iloc[:, :-dy].values, dtype=float),
                dataset.x.detach().cpu().numpy())
        )


if __name__ == '__main__':
    unittest.main()

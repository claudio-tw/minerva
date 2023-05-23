import unittest
import numpy as np
from scipy.stats import norm

from minerva import normalize


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
        self.assertTrue(
            np.allclose(
                normalized_samples,
                exact_cdf,
                rtol=5e-2,
                atol=1e-3,
            )
        )


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import pandas as pd

from minerva.iterable_dataset import MyDataset


class IterableDatasetTest(unittest.TestCase):
    def test_mydataset(self):
        n = 1000
        df = 10
        dy = 1
        float_features = [f'f{n}' for n in range(df)]
        # This version of minerva does not support embedding of categorial features yet
        cat_features = []
        target_names = [f'y{n}' for n in range(dy)]
        x = pd.DataFrame(
            np.random.uniform(size=(n, df)),
            columns=float_features
        )
        y = pd.DataFrame(
            np.random.uniform(size=(n, dy)),
            columns=target_names
        )
        data = pd.concat((x, y), axis=1)
        mydataset = MyDataset(
            data,
            float_features,
            cat_features,
            target_names
        )
        self.assertTrue(
            np.allclose(
                x.values,
                mydataset.x.detach().cpu().numpy()
            )
        )
        self.assertTrue(
            np.allclose(
                y.values,
                mydataset.y.detach().cpu().numpy()
            )
        )


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import pandas as pd
import torch
from minerva.embedder import Embedder
from minerva.iterable_dataset import MyDataset


class EmbedderTest(unittest.TestCase):
    def test_no_categorical(self):
        embedder = Embedder(0, [], 1)
        n, d = 10, 3
        x = torch.randn((n, d))
        xe = embedder(x).reshape(n, d)
        x_ = x.detach().cpu().numpy()
        xe_ = xe.detach().cpu().numpy()
        self.assertTrue(
            np.allclose(x_, xe_)
        )

    def test_cat_and_cont(self):
        n = 100
        n_cat_features = 2
        n_cont_features = 3
        d = n_cat_features + n_cont_features
        feat_sizes = [5, 5]
        xcat = torch.randint(5, (n, n_cat_features))
        xcont = torch.randn((n, n_cont_features))
        x = torch.cat([xcat, xcont], dim=1)
        embedder = Embedder(n_cat_features, feat_sizes, 3)
        xe = embedder(x)
        xcont_ = xcont.detach().cpu().numpy()
        xecont_ = xe[:, n_cat_features:, 0].detach().cpu().numpy()
        self.assertTrue(
            np.allclose(xcont_, xecont_)
        )

    def test_from_dataset(self):
        n = 100
        n_cat_features = 2
        n_cont_features = 3
        n_features = n_cat_features + n_cont_features
        d = n_cat_features + n_cont_features
        feat_sizes = [5, 5]

        xcat = np.random.randint(low=0, high=5, size=(n, n_cat_features))
        xcont = np.random.uniform(size=(n, n_cont_features))
        x = np.concatenate([xcat, xcont], axis=1)
        y = np.random.uniform(size=(n, 1))
        data = np.concatenate([x, y], axis=1)
        float_features = [f'f{n}' for n in range(n_cat_features, n_features)]
        cat_features = [f'f{n}' for n in range(n_cat_features)]
        targets = ['y']
        df = pd.DataFrame(data, columns=cat_features +
                          float_features + targets)
        dataset = MyDataset(
            df=df,
            float_features=float_features,
            cat_features=cat_features,
            targets=targets
        )
        embedder = Embedder(n_cat_features, feat_sizes, 3)
        xe = embedder(dataset.x)
        xcont_ = xcont
        xecont_ = xe[:, n_cat_features:, 0].detach().cpu().numpy()
        self.assertTrue(
            np.allclose(xcont_, xecont_)
        )


if __name__ == '__main__':
    unittest.main()

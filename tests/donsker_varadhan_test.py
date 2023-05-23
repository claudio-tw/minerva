import unittest
import torch
from minerva import donsker_varadhan as dv


class DonskerVaradhanTest(unittest.TestCase):
    def test_objective_functions(self):
        n: int = 500
        d: int = 32
        x = torch.randn(size=(n, d))
        y = torch.randn(size=(n, d))
        f = dv.TestFunction(d)
        v = dv.v(x, y, f)
        self.assertTrue(isinstance(v, torch.Tensor))
        self.assertFalse(torch.isnan(v))


if __name__ == '__main__':
    unittest.main()

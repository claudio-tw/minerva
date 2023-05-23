import torch
from torch import nn
from torch import Tensor


class TestFunction(nn.Module):
    def __init__(self, d: int, dtype=torch.float32):
        super(TestFunction, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(d, 512, dtype=dtype),
            nn.LeakyReLU(),
            nn.Linear(512, 64, dtype=dtype),
            nn.ELU(),
            nn.Linear(64, 1, dtype=dtype),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.f(x)


def v(
        x: Tensor,
        y: Tensor,
        f: TestFunction,
) -> Tensor:
    """
    Computes the argument of the supremum in the Donsker-Varadhan representation
    of the Kullback-Leibner divergence D(P||Q)):
    E_P[f] - log(E_Q[exp(f)]).

    `x` is assumed to store iid samples from P
    `y` is assumed to store iid samples from Q
    `f` is the sequential neural network that parametrises the L^1_P function f
    """
    return f(x).mean() - torch.log(torch.exp(f(y)).mean())

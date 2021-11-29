# Rodrigo C. Daudt
# rcdaudt.github.io

from torch import exp, abs
from torch.nn.modules.loss import _Loss
from torch import Tensor

class GaussianNLLLoss(_Loss):
    """Gaussian negative log-likelihood loss using log-variance

    For more details see:
        Kendall, Alex, and Yarin Gal. "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?."
        Advances in Neural Information Processing Systems 30 (2017): 5574-5584.

    Adapted from:
        torch.nn.GaussianNLLLoss: https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html#torch.nn.GaussianNLLLoss
        torch.nn.functional.gaussian_nll_loss: https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
    """
    __constants__ = ['full', 'eps', 'max_clamp', 'reduction']
    full: bool
    eps: float
    max_clamp: float

    def __init__(self, *, full: bool = False, max_clamp: float = 10.0, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super(GaussianNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.max_clamp = max_clamp # for exponential stability 
        self.eps = eps # for division stability

        if self.full:
            print('FULL GAUSSIAN NLL LOSS NOT YET IMPLEMENTED')
            raise NotImplementedError

    def forward(self, input: Tensor, target: Tensor, log_var: Tensor) -> Tensor:
        loss =  0.5 * (log_var + (input - target)**2 / (exp(log_var.clamp(max=self.max_clamp)) + self.eps))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LaplacianNLLLoss(_Loss):
    """Laplacian negative log-likelihood loss using log-variance

    For more details see:
        I don't know yet

    Adapted from:
        torch.nn.GaussianNLLLoss: https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html#torch.nn.GaussianNLLLoss
        torch.nn.functional.gaussian_nll_loss: https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
    """
    __constants__ = ['full', 'eps', 'max_clamp', 'reduction']
    full: bool
    eps: float
    max_clamp: float

    def __init__(self, *, full: bool = False, max_clamp: float = 10.0, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super(LaplacianNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.max_clamp = max_clamp # for exponential stability 
        self.eps = eps # for division stability

        if self.full:
            print('FULL LAPLACIAN NLL LOSS NOT YET IMPLEMENTED')
            raise NotImplementedError

    def forward(self, input: Tensor, target: Tensor, log_var: Tensor) -> Tensor:
        loss =  0.5 * (log_var + abs(input - target) / (exp(log_var.clamp(max=self.max_clamp)) + self.eps))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

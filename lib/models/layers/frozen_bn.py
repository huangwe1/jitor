import jittor as jt
from jittor import nn


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    """

    def __init__(self, n):
        super().__init__()
        self.weight = jt.ones(n)
        self.bias = jt.zeros(n)
        self.running_mean = jt.zeros(n)
        self.running_var = jt.ones(n)
        self.weight.stop_grad()
        self.bias.stop_grad()
        self.running_mean.stop_grad()
        self.running_var.stop_grad()

    def execute(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).pow(-0.5)
        bias = b - rm * scale
        return x * scale + bias

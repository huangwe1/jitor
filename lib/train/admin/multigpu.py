from jittor import nn


def is_multi_gpu(net):
    return False


class MultiGPU(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def execute(self, *args, **kwargs):
        return self.module(*args, **kwargs)

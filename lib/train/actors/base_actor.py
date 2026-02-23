from lib.utils import TensorDict


class BaseActor:
    """ Base class for actor. """
    def __init__(self, net, objective):
        self.net = net
        self.objective = objective

    def __call__(self, data: TensorDict):
        raise NotImplementedError

    def to(self, device):
        pass

    def train(self, mode=True):
        if mode:
            self.net.train()
        else:
            self.net.eval()

    def eval(self):
        self.train(False)

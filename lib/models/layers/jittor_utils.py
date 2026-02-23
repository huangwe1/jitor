"""
Jittor replacements for timm utility functions.
Provides: Mlp, DropPath, trunc_normal_, lecun_normal_, to_2tuple, named_apply
"""
import math
import jittor as jt
from jittor import nn


def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def trunc_normal_(var, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization (in-place)."""
    with jt.no_grad():
        l = (1. + math.erf((a - mean) / (std * math.sqrt(2.)))) / 2.
        u = (1. + math.erf((b - mean) / (std * math.sqrt(2.)))) / 2.
        val = jt.init.uniform(var.shape, dtype=var.dtype, low=2 * l - 1, high=2 * u - 1)
        val = jt.erfinv(val)
        val = val * (std * math.sqrt(2.)) + mean
        val = jt.clamp(val, min_v=a, max_v=b)
        var.assign(val)
    return var


def lecun_normal_(var):
    fan_in = var.shape[1] if var.ndim > 1 else var.shape[0]
    std = math.sqrt(1.0 / fan_in)
    trunc_normal_(var, std=std)
    return var


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        if self.drop_prob == 0. or not self.is_training():
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = jt.rand(shape)
        random_tensor = jt.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def named_apply(fn, module, name='', depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name,
                    depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def adapt_input_conv(in_chans, conv_weight):
    """Adapt input conv weight for different input channel counts."""
    conv_type = conv_weight.dtype
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        conv_weight = conv_weight.sum(dim=1, keepdims=True)
    elif in_chans != 3:
        repeat = int(math.ceil(in_chans / 3))
        conv_weight = jt.concat([conv_weight] * repeat, dim=1)[:, :in_chans, :, :]
        conv_weight = conv_weight * (3 / float(in_chans))
    return conv_weight

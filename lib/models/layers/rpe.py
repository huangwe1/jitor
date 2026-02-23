import jittor as jt
from jittor import nn
from lib.models.layers.jittor_utils import trunc_normal_


def generate_2d_relative_positional_encoding_index(z_shape, x_shape):
    z_2d_index_h, z_2d_index_w = jt.meshgrid(jt.arange(z_shape[0]), jt.arange(z_shape[1]))

    x_2d_index_h, x_2d_index_w = jt.meshgrid(jt.arange(x_shape[0]), jt.arange(x_shape[1]))

    z_2d_index_h = z_2d_index_h.flatten(0)
    z_2d_index_w = z_2d_index_w.flatten(0)
    x_2d_index_h = x_2d_index_h.flatten(0)
    x_2d_index_w = x_2d_index_w.flatten(0)

    diff_h = z_2d_index_h.unsqueeze(1) - x_2d_index_h.unsqueeze(0)
    diff_w = z_2d_index_w.unsqueeze(1) - x_2d_index_w.unsqueeze(0)

    diff = jt.stack((diff_h, diff_w), dim=-1)
    _, indices = jt.unique(diff.view(-1, 2), return_inverse=True, dim=0)
    return indices.view(z_shape[0] * z_shape[1], x_shape[0] * x_shape[1])


def generate_2d_concatenated_self_attention_relative_positional_encoding_index(z_shape, x_shape):
    z_2d_index_h, z_2d_index_w = jt.meshgrid(jt.arange(z_shape[0]), jt.arange(z_shape[1]))
    x_2d_index_h, x_2d_index_w = jt.meshgrid(jt.arange(x_shape[0]), jt.arange(x_shape[1]))

    z_2d_index_h = z_2d_index_h.flatten(0)
    z_2d_index_w = z_2d_index_w.flatten(0)
    x_2d_index_h = x_2d_index_h.flatten(0)
    x_2d_index_w = x_2d_index_w.flatten(0)

    concatenated_2d_index_h = jt.concat((z_2d_index_h, x_2d_index_h))
    concatenated_2d_index_w = jt.concat((z_2d_index_w, x_2d_index_w))

    diff_h = concatenated_2d_index_h.unsqueeze(1) - concatenated_2d_index_h.unsqueeze(0)
    diff_w = concatenated_2d_index_w.unsqueeze(1) - concatenated_2d_index_w.unsqueeze(0)

    z_len = z_shape[0] * z_shape[1]
    x_len = x_shape[0] * x_shape[1]
    a = jt.zeros((z_len + x_len,), dtype=jt.int64)
    a[z_len:] = 1
    b = a.unsqueeze(1).repeat(1, z_len + x_len)
    c = a.unsqueeze(0).repeat(z_len + x_len, 1)

    diff = jt.stack((diff_h, diff_w, b, c), dim=-1)
    _, indices = jt.unique(diff.view((z_len + x_len) * (z_len + x_len), 4), return_inverse=True, dim=0)
    return indices.view((z_len + x_len), (z_len + x_len))


def generate_2d_concatenated_cross_attention_relative_positional_encoding_index(z_shape, x_shape):
    z_2d_index_h, z_2d_index_w = jt.meshgrid(jt.arange(z_shape[0]), jt.arange(z_shape[1]))
    x_2d_index_h, x_2d_index_w = jt.meshgrid(jt.arange(x_shape[0]), jt.arange(x_shape[1]))

    z_2d_index_h = z_2d_index_h.flatten(0)
    z_2d_index_w = z_2d_index_w.flatten(0)
    x_2d_index_h = x_2d_index_h.flatten(0)
    x_2d_index_w = x_2d_index_w.flatten(0)

    concatenated_2d_index_h = jt.concat((z_2d_index_h, x_2d_index_h))
    concatenated_2d_index_w = jt.concat((z_2d_index_w, x_2d_index_w))

    diff_h = x_2d_index_h.unsqueeze(1) - concatenated_2d_index_h.unsqueeze(0)
    diff_w = x_2d_index_w.unsqueeze(1) - concatenated_2d_index_w.unsqueeze(0)

    z_len = z_shape[0] * z_shape[1]
    x_len = x_shape[0] * x_shape[1]

    a = jt.zeros(z_len + x_len, dtype=jt.int64)
    a[z_len:] = 1
    c = a.unsqueeze(0).repeat(x_len, 1)

    diff = jt.stack((diff_h, diff_w, c), dim=-1)
    _, indices = jt.unique(diff.view(x_len * (z_len + x_len), 3), return_inverse=True, dim=0)
    return indices.view(x_len, (z_len + x_len))


class RelativePosition2DEncoder(nn.Module):
    def __init__(self, num_heads, embed_size):
        super().__init__()
        self.relative_position_bias_table = jt.zeros((num_heads, embed_size))
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def execute(self, attn_rpe_index):
        return self.relative_position_bias_table[:, attn_rpe_index].unsqueeze(0)

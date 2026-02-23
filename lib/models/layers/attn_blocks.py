import math
import jittor as jt
from jittor import nn
from lib.models.layers.jittor_utils import Mlp, DropPath
from lib.models.layers.attn import Attention


def candidate_elimination(attn, tokens, lens_t, keep_ratio, global_index, box_mask_z):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    """
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        box_mask_z_expanded = box_mask_z.unsqueeze(1).unsqueeze(-1).expand([-1, attn_t.shape[1], -1, attn_t.shape[-1]])
        attn_t = attn_t * box_mask_z_expanded.float()
        attn_t = attn_t.sum(dim=2) / (box_mask_z_expanded.float().sum(dim=2) + 1e-8)
        attn_t = attn_t.mean(dim=1)  # B, L_s
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    sorted_indices = jt.argsort(attn_t, dim=1, descending=True)[0]
    topk_idx = sorted_indices[:, :lens_keep]
    non_topk_idx = sorted_indices[:, lens_keep:]

    keep_index = jt.gather(global_index, 1, topk_idx)
    removed_index = jt.gather(global_index, 1, non_topk_idx)

    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    B, L, C = tokens_s.shape
    attentive_tokens = jt.gather(tokens_s, 1, topk_idx.unsqueeze(-1).expand([-1, -1, C]))

    tokens_new = jt.concat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index


class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search

    def execute(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None):
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        x = x + self.drop_path(x_attn)
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(
                attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_search, removed_index_search, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def execute(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

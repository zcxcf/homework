from functools import partial

import torch
import torch.nn as nn
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

import torch
import torch.nn as nn
from functools import partial
from timm.layers import PatchEmbed, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
from timm.models.vision_transformer import Attention

def to_2tuple(x):
    """ Converts a scalar or tuple into a 2-tuple. """
    if isinstance(x, tuple):
        return x
    return (x, x)

def generate_mask(length: int, n: int):
    mask = torch.cat([torch.ones(n), torch.zeros(length - n)])
    return mask

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x, mask=None):
        x = self.fc1(x)
        x = self.act(x)
        if mask is not None:
            x = x * mask
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MatBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            init_values=0,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer=None
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, mask=None):
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x, mask)))
        return x

class MatVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, depth, **kwargs):
        super(MatVisionTransformer, self).__init__(block_fn=MatBlock, **kwargs)
        self.depth = depth
    def forward_features(self, x, mask=None):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i in range(self.depth):
            x = self.blocks[i](x, mask)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = self.forward_features(x, mask)
        x = self.forward_head(x)
        return x


class VisionTransformerEx(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        super(VisionTransformerEx, self).__init__(**kwargs)


if __name__ == '__main__':
    # model = MatVisionTransformer(
    #     patch_size=16, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4, qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6))
    # mask = generate_mask(192*4, 192).to('cuda')
    # model = model.to('cuda')
    # src = torch.rand((1, 3, 224, 224))
    # src = src.to('cuda')
    # out = model(src, mask)
    # print(out.shape)
    # print()

    model = VisionTransformerEx(
        patch_size=16, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    print(model)
    mask = generate_mask(192 * 4, 192).to('cuda')
    model = model.to('cuda')
    src = torch.rand((1, 3, 224, 224))
    src = src.to('cuda')
    out = model(src)
    print(out.shape)
    print()

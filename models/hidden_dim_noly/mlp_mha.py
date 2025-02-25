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
from timm.layers import Mlp as VitMlp
from timm.models.vision_transformer import Attention

from ptflops import get_model_complexity_info
from functools import partial
from timm.layers import Mlp as VitMlp
from timm.models.vision_transformer import Attention

from ptflops import get_model_complexity_info
from torch.jit import Final
# from utils.initial import init_v2
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType


class ModifiedVitMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            embed_dim=192,
            mlp_ratio=4,
            act_layer=nn.GELU,
            scale_factors=None,
    ):
        super().__init__()
        if scale_factors is None:
            scale_factors = [1 / 4, 1 / 2, 1]
        self.scale_factors = scale_factors  # List of scale factors for 's', 'm', 'l', 'xl'

        in_features = embed_dim
        out_features = embed_dim
        self.hidden_features = embed_dim*mlp_ratio

        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, self.hidden_features)
        self.act = act_layer()
        self.fc2 = linear_layer(self.hidden_features, out_features)


    def forward(self, x):
        if self.current_subset_hd is None:
            raise ValueError("Subnetwork size not configured. Call `configure_subnetwork` first.")

        self.up_proj = self.fc1.weight[:self.current_subset_hd]
        self.up_bias = self.fc1.bias[:self.current_subset_hd]

        self.down_proj = self.fc2.weight[:, :self.current_subset_hd]
        self.down_bias = self.fc2.bias

        x_middle = F.linear(x, self.up_proj, bias=self.up_bias)

        down_proj = F.linear(self.act(x_middle), self.down_proj, bias=self.down_bias)

        self.current_subset_hd = None

        return down_proj

    def configure_subnetwork(self, flag):
        """Configure subnetwork size based on flag."""
        hd = self.hidden_features
        if flag == 's':
            scale = self.scale_factors[0]
        elif flag == 'm':
            scale = self.scale_factors[1]
        else:
            scale = self.scale_factors[2]

        self.current_subset_hd = int(hd * scale)


class ModifiedAttention(nn.Module):
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            scale_factors=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if scale_factors is None:
            scale_factors = [1 / 4, 1 / 2, 1]
        self.scale_factors = scale_factors  # List of scale factors for 's', 'm', 'l'
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads

        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.current_subset_dim is None:
            raise ValueError("Subnetwork size not configured. Call `configure_subnetwork` first.")
        sub_input = x

        num_heads = int(self.num_heads*self.dim_scale)

        B, N, C = sub_input.shape

        self.sub_qkv_weight = torch.cat((self.qkv.weight[:self.current_subset_dim],
                                         self.qkv.weight[768:768+self.current_subset_dim],
                                         self.qkv.weight[1536:1536+ self.current_subset_dim]), dim=0)

        self.sub_qkv_bias = torch.cat((self.qkv.bias[:self.current_subset_dim],
                                       self.qkv.bias[768:768+self.current_subset_dim],
                                       self.qkv.bias[1536:1536+self.current_subset_dim]), dim=0)

        qkv_out = F.linear(sub_input, self.sub_qkv_weight, bias=self.sub_qkv_bias)

        qkv = qkv_out.reshape(B, N, 3, num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.fused_attn:
            proj_input = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            proj_input = attn @ v

        qkv_output = proj_input.transpose(1, 2).reshape(B, N, self.current_subset_dim)

        self.sub_proj_weight = self.proj.weight[:, :self.current_subset_dim]
        self.sub_proj_bias = self.proj.bias

        proj_output = F.linear(qkv_output, self.sub_proj_weight, bias=self.sub_proj_bias)

        output = self.proj_drop(proj_output)

        self.current_subset_dim = None

        return output

    def configure_subnetwork(self, flag):
        """Configure subnetwork size based on flag."""
        dim = self.dim
        if flag == 's':
            self.dim_scale = self.scale_factors[0]
        elif flag == 'm':
            self.dim_scale = self.scale_factors[1]
        else:
            self.dim_scale = self.scale_factors[2]

        self.current_subset_dim = int(dim * self.dim_scale)


class MatVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, embed_dim, mlp_ratio, depth, num_heads, qkv_bias, **kwargs):
        super(MatVisionTransformer, self).__init__(embed_dim=embed_dim, mlp_ratio=mlp_ratio, depth=depth, **kwargs)
        self.depth = depth
        scale_factors = [1/4, 1/2, 1]  # s, m, l

        # Replace FFN in each layer with ModifiedFFN
        for layer_idx in range(self.depth):
            self.blocks[layer_idx].mlp = ModifiedVitMlp(
                embed_dim=embed_dim,
                mlp_ratio=int(mlp_ratio),
                scale_factors=scale_factors,
            )
            self.blocks[layer_idx].attn = ModifiedAttention(
                dim=embed_dim,
                num_heads=num_heads,
                scale_factors=scale_factors,
                qkv_bias=qkv_bias
            )

    def configure_subnetwork(self, flag):
        """Configure the subnetwork for all layers based on the flag."""
        for layer_idx in range(self.depth):
            self.blocks[layer_idx].mlp.configure_subnetwork(flag)
            self.blocks[layer_idx].attn.configure_subnetwork(flag)

if __name__ == '__main__':
    model = MatVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=100)
    flag = 'l'
    model.eval()
    with torch.no_grad():
        model.configure_subnetwork(flag)
    # print(model)
    # print(model.depth)

    model = model.to('cuda:7')
    src = torch.rand((1, 3, 224, 224))
    src = src.to('cuda:7')
    out = model(src)
    print(out.shape)
    print()

    # with torch.cuda.device(0):
    #     flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    #     print(f"FLOPs: {flops}")
    #     print(f"Params: {params}")


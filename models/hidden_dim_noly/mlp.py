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
            scale_factors = [1 / 8, 1 / 4, 1 / 2, 1]
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

        up_proj = self.fc1.weight[:self.current_subset_hd]
        up_bias = self.fc1.bias[:self.current_subset_hd]

        down_proj = self.fc2.weight[:, :self.current_subset_hd]
        down_bias = self.fc2.bias

        x_middle = F.linear(x, up_proj, bias=up_bias)

        down_proj = F.linear(self.act(x_middle), down_proj, bias=down_bias)

        # x_middle = F.linear(x, self.fc1.weight[:self.current_subset_hd], bias=self.fc1.bias[:self.current_subset_hd])
        #
        # down_proj = F.linear(self.act(x_middle), self.fc2.weight[:, :self.current_subset_hd], bias=self.fc2.bias)


        # x_middle = F.linear(x, self.fc1.weight, bias=self.fc1.bias)
        #
        # down_proj = F.linear(self.act(x_middle), self.fc2.weight, bias=self.fc2.bias)

        # down_proj = self.fc2(self.act(self.fc1(x)))

        # self.current_subset_hd = None

        return down_proj

    def configure_subnetwork(self, flag):
        """Configure subnetwork size based on flag."""
        hd = self.hidden_features
        if flag == 's':
            scale = self.scale_factors[0]  # hd/8
        elif flag == 'm':
            scale = self.scale_factors[1]  # hd/4
        elif flag == 'l':
            scale = self.scale_factors[2]  # hd/2
        else:  # 'xl'
            scale = self.scale_factors[3]  # hd

        self.current_subset_hd = int(hd * scale)


class MatVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, embed_dim, mlp_ratio, depth, **kwargs):
        super(MatVisionTransformer, self).__init__(embed_dim=embed_dim, mlp_ratio=mlp_ratio, depth=depth, **kwargs)
        self.depth = depth
        scale_factors = [1/8, 1/4, 1/2, 1]  # s, m, l, xl

        # Replace FFN in each layer with ModifiedFFN
        for layer_idx in range(self.depth):
            self.blocks[layer_idx].mlp = ModifiedVitMlp(
                embed_dim=embed_dim,
                mlp_ratio=int(mlp_ratio),
                scale_factors=scale_factors,
            )

    def configure_subnetwork(self, flag):
        """Configure the subnetwork for all layers based on the flag."""

        if isinstance(flag, list):
            for layer_idx, f in enumerate(flag):
                self.blocks[layer_idx].mlp.configure_subnetwork(f)
        else:
            for layer_idx in range(self.depth):
                self.blocks[layer_idx].mlp.configure_subnetwork(flag)



if __name__ == '__main__':
    model = MatVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
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


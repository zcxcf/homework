from functools import partial

import torch
import torch.nn as nn
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
import torch
import torch.nn as nn
from functools import partial
from timm.layers import Mlp as VitMlp
from timm.models.vision_transformer import Attention

from ptflops import get_model_complexity_info
from torch.jit import Final
from utils.initial import init_v2


def to_2tuple(x):
    """ Converts a scalar or tuple into a 2-tuple. """
    if isinstance(x, tuple):
        return x
    return (x, x)


class ModifiedLN(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            embed_dim=192,
            scale_factors=None,
    ):
        super().__init__()
        if scale_factors is None:
            scale_factors = [1 / 4, 1 / 2, 1]
        self.scale_factors = scale_factors  # List of scale factors for 's', 'm', 'l'
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.zeros(embed_dim, ))
        self.bias = nn.Parameter(torch.zeros(embed_dim, ))

    def forward(self, x):
        if self.current_subset_dim is None:
            raise ValueError("Subnetwork size not configured. Call `configure_subnetwork` first.")
        sub_input = x
        self.sub_weight = self.weight[:self.current_subset_dim]
        self.sub_bias = self.bias[:self.current_subset_dim]

        output = F.layer_norm(sub_input, (self.current_subset_dim,), self.sub_weight, self.sub_bias)
        # output = torch.cat((output, x[:, :, self.current_subset_dim:]), dim=2)

        self.current_subset_dim = None

        return output

    def expand_subnetwork_fpi(self):
        if self.current_subset_dim is None:
            raise ValueError("Subnetwork size not configured. Call `configure_subnetwork` first.")
        self.weight[self.current_subset_dim:self.current_subset_dim * 2] = self.weight[:self.current_subset_dim]
        self.bias[self.current_subset_dim:self.current_subset_dim * 2] = self.bias[:self.current_subset_dim]

    def configure_subnetwork(self, dim):
        self.current_subset_dim = dim


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

        num_heads = int(self.current_subset_dim/64)

        B, N, C = sub_input.shape

        self.sub_qkv_weight = torch.cat((self.qkv.weight[:self.current_subset_dim, :self.current_subset_dim],
                                         self.qkv.weight[768:768+self.current_subset_dim, :self.current_subset_dim],
                                         self.qkv.weight[1536:1536+ self.current_subset_dim, :self.current_subset_dim]), dim=0)

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

        qkv_output = proj_input.transpose(1, 2).reshape(B, N, C)
        self.sub_proj_weight = self.proj.weight[:self.current_subset_dim, :self.current_subset_dim]
        self.sub_proj_bias = self.proj.bias[:self.current_subset_dim]
        proj_output = F.linear(qkv_output, self.sub_proj_weight, bias=self.sub_proj_bias)
        output = self.proj_drop(proj_output)
        self.current_subset_dim = None

        return output

    def expand_subnetwork_fpi(self, if_div):
        if self.current_subset_dim is None:
            raise ValueError("Subnetwork size not configured. Call `configure_subnetwork` first.")
        r = int(1 / self.dim_scale)

        self.qkv.weight[:, self.current_subset_dim:self.current_subset_dim * 2] = self.qkv.weight[:,
                                                                                  :self.current_subset_dim]
        if if_div:
            self.qkv.weight /= 2
        self.qkv.weight[self.current_subset_dim:self.current_subset_dim * 2] = self.qkv.weight[:self.current_subset_dim]
        self.qkv.weight[(r + 1) * self.current_subset_dim:(r + 2) * self.current_subset_dim] = self.qkv.weight[
                                                                                               r * self.current_subset_dim:(
                                                                                                                                       r + 1) * self.current_subset_dim]
        self.qkv.weight[(r * 2 + 1) * self.current_subset_dim:(r * 2 + 2) * self.current_subset_dim] = self.qkv.weight[
                                                                                                       r * 2 * self.current_subset_dim:(
                                                                                                                                                   r * 2 + 1) * self.current_subset_dim]
        self.qkv.bias[self.current_subset_dim:self.current_subset_dim * 2] = self.qkv.bias[:self.current_subset_dim]
        self.qkv.bias[(r + 1) * self.current_subset_dim:(r + 2) * self.current_subset_dim] = self.qkv.bias[
                                                                                             r * self.current_subset_dim:(
                                                                                                                                     r + 1) * self.current_subset_dim]
        self.qkv.bias[(r * 2 + 1) * self.current_subset_dim:(r * 2 + 2) * self.current_subset_dim] = self.qkv.bias[
                                                                                                     r * 2 * self.current_subset_dim:(
                                                                                                                                                 r * 2 + 1) * self.current_subset_dim]

        self.proj.weight[:, self.current_subset_dim:self.current_subset_dim * 2] = self.proj.weight[:,
                                                                                   :self.current_subset_dim]
        if if_div:
            self.proj.weight /= 2
        self.proj.weight[self.current_subset_dim:self.current_subset_dim * 2] = self.proj.weight[
                                                                                :self.current_subset_dim]

        self.proj.bias[self.current_subset_dim:self.current_subset_dim * 2] = self.proj.bias[:self.current_subset_dim]

    def configure_subnetwork(self, dim):
        self.dim_scale = 64/dim
        self.current_subset_dim = dim


class ModifiedVitMlp(nn.Module):
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
        self.scale_factors = scale_factors  # List of scale factors for 's', 'm', 'l'
        self.embed_dim = embed_dim
        in_features = embed_dim
        out_features = embed_dim
        self.hidden_features = embed_dim * mlp_ratio

        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, self.hidden_features)
        self.act = act_layer()
        self.fc2 = linear_layer(self.hidden_features, out_features)

    def forward(self, x):
        if self.current_subset_dim is None:
            raise ValueError("Subnetwork size not configured. Call `configure_subnetwork` first.")

        sub_input = x

        self.up_proj = self.fc1.weight[:self.current_subset_dim * 4, :self.current_subset_dim]
        self.up_bias = self.fc1.bias[:self.current_subset_dim * 4]

        self.down_proj = self.fc2.weight[:self.current_subset_dim, :self.current_subset_dim * 4]
        self.down_bias = self.fc2.bias[:self.current_subset_dim]

        x_middle = F.linear(sub_input, self.up_proj, bias=self.up_bias)

        output = F.linear(self.act(x_middle), self.down_proj, bias=self.down_bias)
        self.current_subset_dim = None

        return output

    def expand_subnetwork_fpi(self, if_div):
        if self.current_subset_dim is None:
            raise ValueError("Subnetwork size not configured. Call `configure_subnetwork` first.")

        self.fc1.weight[:, self.current_subset_dim:self.current_subset_dim * 2] = self.fc1.weight[:,
                                                                                  :self.current_subset_dim]
        if if_div:
            self.fc1.weight /= 2
        self.fc1.weight[self.current_subset_dim * 4:self.current_subset_dim * 4 * 2] = self.fc1.weight[
                                                                                       :self.current_subset_dim * 4]

        self.fc1.bias[self.current_subset_dim * 4:self.current_subset_dim * 4 * 2] = self.fc1.bias[
                                                                                     :self.current_subset_dim * 4]

        self.fc2.weight[:, self.current_subset_dim * 4:self.current_subset_dim * 4 * 2] = self.fc2.weight[:,
                                                                                          :self.current_subset_dim * 4]
        if if_div:
            self.fc2.weight /= 2
        self.fc2.weight[self.current_subset_dim:self.current_subset_dim * 2] = self.fc2.weight[:self.current_subset_dim]

        self.fc2.bias[self.current_subset_dim:self.current_subset_dim * 2] = self.fc2.bias[:self.current_subset_dim]

    def configure_subnetwork(self, dim):
        self.current_subset_dim = dim


class ModifiedHead(nn.Module):
    def __init__(
            self,
            embed_dim=192,
            num_classes=100,
            scale_factors=None,
    ):
        super().__init__()
        if scale_factors is None:
            scale_factors = [1 / 4, 1 / 2, 1]
        self.scale_factors = scale_factors  # List of scale factors for 's', 'm', 'l'
        self.embed_dim = embed_dim

        self.weight = nn.Parameter(torch.zeros(num_classes, embed_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes, ))

    def forward(self, x):
        if self.current_subset_dim is None:
            raise ValueError("Subnetwork size not configured. Call `configure_subnetwork` first.")

        sub_input = x

        self.sub_weight = self.weight[:, :self.current_subset_dim]
        self.sub_bias = self.bias

        output = F.linear(sub_input, self.sub_weight, bias=self.sub_bias)

        self.current_subset_dim = None

        return output

    def expand_subnetwork_fpi(self, if_div):
        if self.current_subset_dim is None:
            raise ValueError("Subnetwork size not configured. Call `configure_subnetwork` first.")

        self.weight[:, self.current_subset_dim:self.current_subset_dim * 2] = self.weight[:, :self.current_subset_dim]
        if if_div:
            self.weight /= 2

    def configure_subnetwork(self, dim):
        self.current_subset_dim = dim


class MatVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, embed_dim, mlp_ratio, depth, num_heads, qkv_bias, num_classes, **kwargs):
        super(MatVisionTransformer, self).__init__(embed_dim=embed_dim, mlp_ratio=mlp_ratio, depth=depth, **kwargs)
        self.depth = depth
        self.scale_factors = [1 / 4, 1 / 2, 1]  # s, m, l
        self.embed_dim = embed_dim

        self.norm = ModifiedLN(
            embed_dim=embed_dim,
            scale_factors=self.scale_factors
        )
        self.head = ModifiedHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            scale_factors=self.scale_factors
        )

        for layer_idx in range(self.depth):
            self.blocks[layer_idx].norm1 = ModifiedLN(
                embed_dim=embed_dim,
                scale_factors=self.scale_factors,
            )
            self.blocks[layer_idx].norm2 = ModifiedLN(
                embed_dim=embed_dim,
                scale_factors=self.scale_factors,
            )
            self.blocks[layer_idx].mlp = ModifiedVitMlp(
                embed_dim=embed_dim,
                scale_factors=self.scale_factors,
            )
            self.blocks[layer_idx].attn = ModifiedAttention(
                dim=embed_dim,
                num_heads=num_heads,
                scale_factors=self.scale_factors,
                qkv_bias=qkv_bias
            )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = x[:, :, :self.sub_dim]
        # print("x", x[0, 0, :10])
        x = self.blocks(x)
        # print("x2", x[0, 0, :10])
        x = self.norm(x)
        return x

    def configure_subnetwork(self, flag):
        """Configure the subnetwork for all layers based on the flag."""
        dim = self.embed_dim
        if flag == 's':
            scale = self.scale_factors[0]  # hd/4
        elif flag == 'm':
            scale = self.scale_factors[1]  # hd/2
        else:
            scale = self.scale_factors[2]  # hd

        self.sub_dim = int(dim * scale)

        self.norm.configure_subnetwork(flag)
        self.head.configure_subnetwork(flag)
        for layer_idx in range(self.depth):
            self.blocks[layer_idx].norm1.configure_subnetwork(flag)
            self.blocks[layer_idx].norm2.configure_subnetwork(flag)
            self.blocks[layer_idx].attn.configure_subnetwork(flag)
            self.blocks[layer_idx].mlp.configure_subnetwork(flag)

    def configure_subnetwork_fined(self, num_head):
        self.sub_dim = int(64 * num_head)

        self.norm.configure_subnetwork(self.sub_dim)
        self.head.configure_subnetwork(self.sub_dim)
        for layer_idx in range(self.depth):
            self.blocks[layer_idx].norm1.configure_subnetwork(self.sub_dim)
            self.blocks[layer_idx].norm2.configure_subnetwork(self.sub_dim)
            self.blocks[layer_idx].attn.configure_subnetwork(self.sub_dim)
            self.blocks[layer_idx].mlp.configure_subnetwork(self.sub_dim)

    def configure_subnetwork_list(self, head_flag, flag_list):
        self.norm.configure_subnetwork(head_flag)
        self.head.configure_subnetwork(head_flag)

        for layer_idx, flag in enumerate(flag_list):
            self.blocks[layer_idx].norm1.configure_subnetwork(flag)
            self.blocks[layer_idx].norm2.configure_subnetwork(flag)
            self.blocks[layer_idx].attn.configure_subnetwork(flag)
            self.blocks[layer_idx].mlp.configure_subnetwork(flag)



    def expand_subnetwork(self, type):
        self.cls_token[:, :, self.sub_dim:self.sub_dim * 2] = self.cls_token[:, :, :self.sub_dim]
        self.pos_embed[:, :, self.sub_dim:self.sub_dim * 2] = self.pos_embed[:, :, :self.sub_dim]
        self.patch_embed.proj.weight[self.sub_dim:self.sub_dim * 2] = self.patch_embed.proj.weight[:self.sub_dim]
        self.patch_embed.proj.bias[self.sub_dim:self.sub_dim * 2] = self.patch_embed.proj.bias[:self.sub_dim]
        self.norm.expand_subnetwork_fpi()

        if type == 'fpi_pre_small':
            self.head.expand_subnetwork_fpi(if_div=False)
            for layer_idx in range(self.depth):
                self.blocks[layer_idx].attn.expand_subnetwork_fpi(if_div=False)
                self.blocks[layer_idx].mlp.expand_subnetwork_fpi(if_div=False)
                self.blocks[layer_idx].norm1.expand_subnetwork_fpi()
                self.blocks[layer_idx].norm2.expand_subnetwork_fpi()

        if type == 'fpi_pre_big':
            self.head.expand_subnetwork_fpi(if_div=True)
            for layer_idx in range(self.depth):
                self.blocks[layer_idx].attn.expand_subnetwork_fpi(if_div=True)
                self.blocks[layer_idx].mlp.expand_subnetwork_fpi(if_div=True)
                self.blocks[layer_idx].norm1.expand_subnetwork_fpi()
                self.blocks[layer_idx].norm2.expand_subnetwork_fpi()


if __name__ == '__main__':
    model = MatVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=100)
    flag = 's'
    model.eval()
    with torch.no_grad():
        model.configure_subnetwork_fined(3)

    check_point_path = '/home/ssd7T/zc_reuse/iccv/pretrained_para/vit_tiny.pth'
    checkpoint = torch.load(check_point_path, map_location='cuda:7')

    init_v2(model, checkpoint, init_width=192, depth=12)

    model = model.to('cuda:7')
    src = torch.rand((1, 3, 224, 224))
    src = src.to('cuda:7')
    out = model(src)
    print(out.shape)
    print(out[0, :10])
    print('-' * 1000)

    with torch.no_grad():
        model.configure_subnetwork_fined(4)
    out = model(src)
    print(out.shape)
    print(out[0, :10])
    print('-' * 1000)

    with torch.no_grad():
        model.configure_subnetwork_fined(7)
    out = model(src)
    print(out.shape)
    print(out[0, :10])
    print('-' * 1000)

    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    #     print(name)
    #
    #     # if "layer" in name:
    #     #     param.requires_grad = True
    #
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name} is trainable")
    #     else:
    #         print(f"{name} is frozen")

    # with torch.no_grad():
    #     model.configure_subnetwork(flag)
    #     model.expand_subnetwork(type='fpi_pre_small')
    # out = model(src)
    # print(out.shape)
    # print(out[0, :10])
    #
    #
    # model = timm.models.vision_transformer.vit_tiny_patch16_224(pretrained=False, num_classes=100)
    # model.to("cuda:7")
    # model_path = '/home/ssd7T/zc_reuse/iccv/pretrained_para/vit_tiny.pth'
    # para = torch.load(model_path, map_location='cuda:7')
    # model.load_state_dict(para)
    #
    # out = model(src)
    # print(out.shape)
    # print(out[0, :10])
    # with torch.cuda.device(0):
    #     flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    #     print(f"FLOPs: {flops}")
    #     print(f"Params: {params}")


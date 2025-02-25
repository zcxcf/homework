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

def _gumbel_sigmoid(
    logits, tau=1, hard=False, training = True, threshold = 0.5
):
    if training :
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # Difference of two` gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else :
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class ModifiedLN(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            embed_dim=192,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(embed_dim, ))
        self.bias = nn.Parameter(torch.zeros(embed_dim, ))

    def forward(self, x):
        sub_input = x
        sub_weight = self.weight[:self.current_subset_dim]
        sub_bias = self.bias[:self.current_subset_dim]

        output = F.layer_norm(sub_input, (self.current_subset_dim,), sub_weight, sub_bias)
        # output = torch.cat((output, x[:, :, self.current_subset_dim:]), dim=2)

        return output

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
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
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
        sub_input = x

        B, N, C = sub_input.shape

        self.sub_qkv_weight = torch.cat((self.qkv.weight[:self.num_heads*self.head_dim, :self.current_subset_dim],
                                         self.qkv.weight[self.dim:self.dim + self.num_heads*self.head_dim, :self.current_subset_dim],
                                         self.qkv.weight[self.dim*2:self.dim*2 + self.num_heads*self.head_dim, :self.current_subset_dim]), dim=0)

        self.sub_qkv_bias = torch.cat((self.qkv.bias[:self.num_heads*self.head_dim],
                                       self.qkv.bias[self.dim:self.dim+self.num_heads*self.head_dim],
                                       self.qkv.bias[self.dim*2:self.dim*2+self.num_heads*self.head_dim]), dim=0)

        qkv_out = F.linear(sub_input, self.sub_qkv_weight, bias=self.sub_qkv_bias)
        qkv = qkv_out.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
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

        qkv_output = proj_input.transpose(1, 2).reshape(B, N, self.num_heads*self.head_dim)

        self.sub_proj_weight = self.proj.weight[:self.current_subset_dim, :self.num_heads*self.head_dim]
        self.sub_proj_bias = self.proj.bias[:self.current_subset_dim]
        proj_output = F.linear(qkv_output, self.sub_proj_weight, bias=self.sub_proj_bias)
        output = self.proj_drop(proj_output)

        return output

    def configure_subnetwork(self, dim, num_heads):
        self.current_subset_dim = dim
        self.num_heads = num_heads


class ModifiedVitMlp(nn.Module):
    def __init__(
            self,
            embed_dim=192,
            mlp_ratio=4,
            act_layer=nn.GELU,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        in_features = embed_dim
        out_features = embed_dim
        self.hidden_features = embed_dim * mlp_ratio

        self.fc1 = nn.Linear(in_features, self.hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, out_features)

        self.router = nn.Linear(1, 3)

    def forward(self, x):
        b, n, d = x.shape

        # print(self.latency.shape)
        # print(self.latency)

        logist = self.router(self.latency).unsqueeze(0)

        # print(logist.shape)
        # print(logist)

        if self.training:
            mask = _gumbel_sigmoid(logist, tau=1, hard=False, training=True)
            self.mask = mask
            mask = torch.cat((mask, mask[:, :, -1:]), dim=-1)

            mask = mask.repeat_interleave(d)
            mask = mask.unsqueeze(0).unsqueeze(0).expand(b, n, 4 * d)

            x_middle = self.act(self.fc1(x)) * mask
            output = self.fc2(x_middle)

        else:
            mask = _gumbel_sigmoid(logist, tau=1, hard=True, training=False)

            self.mask = mask
            mask = torch.cat((mask, mask[:, :, -1:]), dim=-1)

            mask = mask.repeat_interleave(d)

            mask = mask.bool()

            sub_input = x

            self.up_proj = self.fc1.weight[mask, :self.current_subset_dim]
            self.up_bias = self.fc1.bias[mask]

            self.down_proj = self.fc2.weight[:self.current_subset_dim, mask]
            self.down_bias = self.fc2.bias[:self.current_subset_dim]

            x_middle = F.linear(sub_input, self.up_proj, bias=self.up_bias)

            output = F.linear(self.act(x_middle), self.down_proj, bias=self.down_bias)

        return output

    def configure_subnetwork(self, dim, ratio, latency):
        self.current_subset_dim = dim
        self.sub_ratio = ratio
        self.latency = latency


class ModifiedHead(nn.Module):
    def __init__(
            self,
            embed_dim=192,
            num_classes=100,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.zeros(num_classes, embed_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes, ))

    def forward(self, x):
        sub_input = x
        self.sub_weight = self.weight[:, :self.current_subset_dim]
        self.sub_bias = self.bias
        output = F.linear(sub_input, self.sub_weight, bias=self.sub_bias)
        return output

    def configure_subnetwork(self, dim):
        self.current_subset_dim = dim


class MatVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, embed_dim, mlp_ratio, depth, num_heads, qkv_bias, num_classes, **kwargs):
        super(MatVisionTransformer, self).__init__(embed_dim=embed_dim, mlp_ratio=mlp_ratio, depth=depth, num_heads=num_heads, **kwargs)
        self.depth = depth
        self.scale_factors = [1 / 4, 1 / 2, 1]  # s, m, l
        self.embed_dim = embed_dim

        self.norm = ModifiedLN(
            embed_dim=embed_dim,
        )
        self.head = ModifiedHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
        )
        for layer_idx in range(self.depth):
            self.blocks[layer_idx].norm1 = ModifiedLN(
                embed_dim=embed_dim,
            )
            self.blocks[layer_idx].norm2 = ModifiedLN(
                embed_dim=embed_dim,
            )
            self.blocks[layer_idx].mlp = ModifiedVitMlp(
                embed_dim=embed_dim,
            )
            self.blocks[layer_idx].attn = ModifiedAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias
            )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = x[:, :, :self.sub_dim]
        # print("x", x[0, 0, :10])
        for index in self.depth_list:
            x = self.blocks[index](x)
        # x = self.blocks(x)
        # print("x2", x[0, 0, :10])
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)

        mean_list = []

        for i in self.depth_list:
            mask = self.blocks[i].mlp.mask
            cost = mask[:, :, 0] + mask[:, :, 1] * 2 + mask[:, :, 2] * 4
            mean_list.append(torch.mean(cost))

        mean_mask = torch.stack(mean_list)
        mean_mask_value = torch.mean(mean_mask)

        # print(mean_list)
        # print(mean_mask_value)

        return x, mean_mask_value

    def configure_latency(self, latency):
        latency = torch.tensor(latency).to(self.blocks[0].mlp.fc1.weight.device).unsqueeze(0).unsqueeze(0)

        for layer_idx in range(self.depth):
            # self.blocks[layer_idx].attn.configure_subnetwork(self.sub_dim, self.mha_head)
            self.blocks[layer_idx].mlp.configure_subnetwork(latency)
            # self.blocks[layer_idx].norm1.configure_subnetwork(self.sub_dim)
            # self.blocks[layer_idx].norm2.configure_subnetwork(self.sub_dim)

    def configure_subnetwork(self, sub_dim, depth_list, mlp_ratio, mha_head, latency):
        self.sub_dim = sub_dim
        self.depth_list = depth_list
        self.mlp_ratio = mlp_ratio
        self.mha_head = mha_head

        latency = torch.tensor(latency).to(self.blocks[0].mlp.fc1.weight.device).unsqueeze(0).unsqueeze(0)

        self.norm.configure_subnetwork(self.sub_dim)
        self.head.configure_subnetwork(self.sub_dim)

        if isinstance(self.mha_head, list) and isinstance(self.mlp_ratio, list):
            for layer_idx, (head, ratio) in enumerate(zip(mha_head, mlp_ratio)):
                self.blocks[layer_idx].norm1.configure_subnetwork(self.sub_dim)
                self.blocks[layer_idx].norm2.configure_subnetwork(self.sub_dim)
                self.blocks[layer_idx].attn.configure_subnetwork(self.sub_dim, head)
                self.blocks[layer_idx].mlp.configure_subnetwork(self.sub_dim, ratio, latency)
        else:
            for layer_idx in range(self.depth):
                self.blocks[layer_idx].attn.configure_subnetwork(self.sub_dim, self.mha_head)
                self.blocks[layer_idx].mlp.configure_subnetwork(self.sub_dim, self.mlp_ratio, latency)
                self.blocks[layer_idx].norm1.configure_subnetwork(self.sub_dim)
                self.blocks[layer_idx].norm2.configure_subnetwork(self.sub_dim)
if __name__ == '__main__':
    model = MatVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=100)

    sub_dim = 384
    depth_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    mlp_ratio = 4
    mha_head = 6

    model.eval()
    model = model.to('cuda:7')

    with torch.no_grad():
        model.configure_subnetwork(sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio, mha_head=mha_head, latency=1.0)

    check_point_path = '/home/ssd7T/zc_reuse/iccv/pretrained_para/vit_tiny.pth'
    checkpoint = torch.load(check_point_path, map_location='cuda:7')

    src = torch.rand((1, 3, 224, 224))
    src = src.to('cuda:7')
    out, mask = model(src)
    print(mask)

    print(out.shape)
    print(out[0, :10])
    print('-' * 1000)

    # out = model(src)
    # print(out.shape)
    # print(out[0, :10])
    # print('-' * 1000)
    #
    # out = model(src)
    # print(out.shape)
    # print(out[0, :10])
    # print('-' * 1000)

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


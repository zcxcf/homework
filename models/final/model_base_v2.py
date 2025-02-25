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
from timm.models.vision_transformer import Block

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
    def __init__(
            self,
            embed_dim=192,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(embed_dim, ))
        self.bias = nn.Parameter(torch.zeros(embed_dim, ))

    def forward(self, x):
        if self.training:
            b, n, d = x.shape
            output = F.layer_norm(x, (d,), self.weight, self.bias)
        else:
            sub_input = x
            b ,n ,d = x.shape
            sub_weight = self.weight[self.dim_bool]
            sub_bias = self.bias[self.dim_bool]
            output = F.layer_norm(sub_input, (d,), sub_weight, sub_bias)
        return output
    def configure_dim_bool(self, dim):
        self.dim_bool = dim


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

        self.router = nn.Linear(1, num_heads)

        self.setted_mask = None
        #     to decide the head 4 5 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sub_input = x
        B, N, C = sub_input.shape
        if self.training:
            mask = self.mask.repeat_interleave(self.head_dim)
            qkv_out = self.qkv(sub_input)
            qkv = qkv_out.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            if self.fused_attn:
                qkv_output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                qkv_output = attn @ v
            proj_input = qkv_output.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
            proj_input = proj_input * mask
            proj_output = self.proj(proj_input)
            output = self.proj_drop(proj_output)
        else:
            mask = self.mask
            if torch.sum(mask) == 0:
                return torch.zeros(x.shape).to(x.device)
            sub_num_heads = (mask == 1).sum().item()
            mask = mask.repeat_interleave(self.head_dim)
            mask_triple = torch.cat((mask, mask, mask), dim=0)
            mask = mask.bool()
            mask_triple = mask_triple.bool()
            self.sub_qkv_weight = self.qkv.weight[mask_triple, :][:, self.dim_bool]
            self.sub_qkv_bias = self.qkv.bias[mask_triple]
            qkv_out = F.linear(sub_input, self.sub_qkv_weight, bias=self.sub_qkv_bias)
            qkv = qkv_out.reshape(B, N, 3, sub_num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            if self.fused_attn:
                qkv_output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                qkv_output = attn @ v
            proj_input = qkv_output.transpose(1, 2).reshape(B, N, sub_num_heads*self.head_dim)
            self.sub_proj_weight = self.proj.weight[self.dim_bool, :][:, mask]
            self.sub_proj_bias = self.proj.bias[self.dim_bool]
            proj_output = F.linear(proj_input, self.sub_proj_weight, bias=self.sub_proj_bias)
            output = self.proj_drop(proj_output)
        return output

    def caculate_mask(self):
        logist = self.router(self.latency)
        if self.training:
            mask = _gumbel_sigmoid(logist, tau=self.tau, hard=False, training=True)
            self.mask = mask
        else:
            if self.setted_mask != None:
                mask = self.setted_mask
            else:
                mask = _gumbel_sigmoid(logist, tau=self.tau, hard=True, training=False)
            self.mask = mask

    def configure_latency(self, latency, tau):
        self.latency = latency
        self.tau = tau

    def configure_dim_bool(self, dim):
        self.dim_bool = dim

    def set_mask(self, mask):
        self.setted_mask = mask


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

        self.router = nn.Linear(1, 4)
        #     ratio 1 2 3 4
        self.setted_mask = None

    def forward(self, x):
        if self.training:
            mask = self.mask
            mask = mask.repeat_interleave(self.embed_dim)
            x_middle = self.act(self.fc1(x)) * mask
            output = self.fc2(x_middle)
        else:
            mask = self.mask
            if torch.sum(mask) == 0:
                return torch.zeros(x.shape).to(x.device)
            mask = mask.repeat_interleave(self.embed_dim)
            mask = mask.bool()
            sub_input = x
            self.up_proj = self.fc1.weight[mask, :][:, self.dim_bool]
            self.up_bias = self.fc1.bias[mask]
            self.down_proj = self.fc2.weight[self.dim_bool, :][:, mask]
            self.down_bias = self.fc2.bias[self.dim_bool]
            x_middle = F.linear(sub_input, self.up_proj, bias=self.up_bias)
            output = F.linear(self.act(x_middle), self.down_proj, bias=self.down_bias)
        return output

    def caculate_mask(self):
        logist = self.router(self.latency)
        if self.training:
            mask = _gumbel_sigmoid(logist, tau=self.tau, hard=False, training=True)
            self.mask = mask
        else:
            if self.setted_mask != None:
                mask = self.setted_mask
            else:
                mask = _gumbel_sigmoid(logist, tau=self.tau, hard=True, training=False)
            self.mask = mask

    def configure_latency(self, latency, tau):
        self.latency = latency
        self.tau = tau

    def configure_dim_bool(self, dim):
        self.dim_bool = dim

    def set_mask(self, mask):
        self.setted_mask = mask

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
        if self.training:
            output = F.linear(x, self.weight, bias=self.bias)
        else:
            sub_input = x
            self.sub_weight = self.weight[:, self.dim_bool]
            self.sub_bias = self.bias
            output = F.linear(sub_input, self.sub_weight, bias=self.sub_bias)
        return output

    def configure_dim_bool(self, dim):
        self.dim_bool = dim

class ModifiedBlock(Block):
    def __init__(self, **kwargs):
        super(ModifiedBlock, self).__init__(**kwargs)

    def forward(self, x, mask_depth_attn, mask_depth_mlp, embed_mask):
        if self.training:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x) * embed_mask))) * mask_depth_attn
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x) * embed_mask))) * mask_depth_mlp
        else:
            if mask_depth_attn == 1:
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            if mask_depth_mlp == 1:
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class MatVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, embed_dim, mlp_ratio, depth, num_heads, qkv_bias, num_classes, block, **kwargs):
        super(MatVisionTransformer, self).__init__(embed_dim=embed_dim, mlp_ratio=mlp_ratio, depth=depth, num_heads=num_heads, block_fn=block, **kwargs)
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
        self.head_dim = embed_dim//num_heads

        self.embed_router = nn.Linear(1, num_heads-1)
        self.depth_attn_router = nn.Linear(1, depth)
        self.depth_mlp_router = nn.Linear(1, depth)

        self.setted_embed_mask = None
        self.setted_depth_attn_mask = None
        self.setted_depth_mlp_mask = None

    def forward_features(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if self.training:
            embed_mask = self.embed_mask
            embed_mask = embed_mask.repeat_interleave(self.head_dim)
            depth_attn_mask = self.depth_attn_mask
            depth_mlp_mask = self.depth_mlp_mask
            for index in range(self.depth):
                x = self.blocks[index](x, depth_attn_mask[index], depth_mlp_mask[index], embed_mask)
            x = self.norm(x) * embed_mask
        else:
            embed_mask = self.embed_mask
            embed_mask = embed_mask.repeat_interleave(self.head_dim)
            embed_mask = embed_mask.bool()
            self.configure_dim_bool(embed_mask)

            depth_attn_mask = self.depth_attn_mask
            depth_mlp_mask = self.depth_mlp_mask

            x = x[:, :, embed_mask]

            for index in range(self.depth):
                x = self.blocks[index](x, depth_attn_mask[index], depth_mlp_mask[index], embed_mask)

            x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        self.caculate_mask_all()

        x = self.forward_features(x)
        x = self.forward_head(x)
        mean_attn_mask_value, mean_mlp_mask_value, mean_embed_dim_mask, mean_depth_attn_mask, mean_depth_mlp_mask = self.caculate_cost()
        return x, mean_attn_mask_value, mean_mlp_mask_value, mean_embed_dim_mask, mean_depth_attn_mask, mean_depth_mlp_mask

    def caculate_cost(self):
        attn_mean_list = []
        mlp_mean_list = []

        for i in range(self.depth):
            mask_attn = self.blocks[i].attn.mask
            mask_mlp = self.blocks[i].mlp.mask

            cost_attn = torch.mean(mask_attn)
            cost_mlp = torch.mean(mask_mlp)

            attn_mean_list.append(torch.mean(cost_attn))
            mlp_mean_list.append(torch.mean(cost_mlp))

        mean_attn_mask = torch.stack(attn_mean_list)
        mean_attn_mask_value = torch.mean(mean_attn_mask)

        mean_mlp_mask = torch.stack(mlp_mean_list)
        mean_mlp_mask_value = torch.mean(mean_mlp_mask)

        mean_embed_dim_mask = torch.mean(self.embed_mask)

        mean_depth_attn_mask = torch.mean(self.depth_attn_mask)
        mean_depth_mlp_mask = torch.mean(self.depth_mlp_mask)

        return mean_attn_mask_value, mean_mlp_mask_value, mean_embed_dim_mask, mean_depth_attn_mask, mean_depth_mlp_mask

    def caculate_mask_all(self):
        embed_logist = self.embed_router(self.latency)
        depth_attn_logist = self.depth_attn_router(self.latency)
        depth_mlp_logist = self.depth_mlp_router(self.latency)
        if self.training:
            embed_mask = _gumbel_sigmoid(embed_logist, tau=1, hard=False, training=True)
            mask_one = torch.ones(1).to(embed_mask.device)
            embed_mask = torch.cat((mask_one, embed_mask), dim=-1)
            self.embed_mask = embed_mask

            depth_attn_mask = _gumbel_sigmoid(depth_attn_logist, tau=1, hard=False, training=True)
            self.depth_attn_mask = depth_attn_mask
            depth_mlp_mask = _gumbel_sigmoid(depth_mlp_logist, tau=1, hard=False, training=True)
            self.depth_mlp_mask = depth_mlp_mask
        else:
            if self.setted_embed_mask != None:
                embed_mask = self.setted_embed_mask
            else:
                embed_mask = _gumbel_sigmoid(embed_logist, tau=5, hard=True, training=False)
            if torch.sum(embed_mask) == 0:
                raise Exception("all zeros")
            mask_one = torch.ones(1).to(embed_mask.device)
            embed_mask = torch.cat((mask_one, embed_mask), dim=-1)
            self.embed_mask = embed_mask

            if self.setted_depth_attn_mask != None:
                depth_attn_mask = self.setted_depth_attn_mask
            else:
                depth_attn_mask = _gumbel_sigmoid(depth_attn_logist, tau=1, hard=True, training=False)
            self.depth_attn_mask = depth_attn_mask
            if self.setted_depth_mlp_mask != None:
                depth_mlp_mask = self.setted_depth_mlp_mask
            else:
                depth_mlp_mask = _gumbel_sigmoid(depth_mlp_logist, tau=1, hard=True, training=False)
            self.depth_mlp_mask = depth_mlp_mask


        for layer_idx in range(self.depth):
            self.blocks[layer_idx].attn.caculate_mask()
            self.blocks[layer_idx].mlp.caculate_mask()

    def configure_latency(self, latency):
        self.latency = latency
        for layer_idx in range(self.depth):
            self.blocks[layer_idx].attn.configure_latency(latency, tau=1)
            self.blocks[layer_idx].mlp.configure_latency(latency, tau=1)

    def configure_dim_bool(self, dim):
        self.norm.configure_dim_bool(dim)
        self.head.configure_dim_bool(dim)

        for layer_idx in range(self.depth):
            self.blocks[layer_idx].attn.configure_dim_bool(dim)
            self.blocks[layer_idx].mlp.configure_dim_bool(dim)
            self.blocks[layer_idx].norm1.configure_dim_bool(dim)
            self.blocks[layer_idx].norm2.configure_dim_bool(dim)

    def set_mask(self, embed_mask, attn_mask, mlp_mask, depth_attn_mask, depth_mlp_mask):
        self.setted_embed_mask = embed_mask
        self.setted_depth_attn_mask = depth_attn_mask
        self.setted_depth_mlp_mask = depth_mlp_mask

        for i in range(self.depth):
            self.blocks[i].attn.set_mask(attn_mask)
            self.blocks[i].mlp.set_mask(mlp_mask)


if __name__ == '__main__':
    model = MatVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=100, block=ModifiedBlock)

    device = "cuda:0"
    latency = torch.rand(1).to(device)

    print(latency)

    model.eval()
    model = model.to(device)
    model.configure_latency(latency=latency)

    # # embed_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
    # embed_mask = torch.tensor([0.0]*6).to(device)
    #
    # mask_attn = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
    # mask_mlp = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(device)
    #
    # depth_attn_mask = torch.tensor([1.0]*12).to(device)
    # depth_mlp_mask = torch.tensor([1.0]*12).to(device)
    #
    # model.set_mask(embed_mask, mask_attn, mask_mlp, depth_attn_mask, depth_mlp_mask)

    #
    # # check_point_path = '/home/ssd7T/zc_reuse/iccv/pretrained_para/vit_tiny.pth'
    # # checkpoint = torch.load(check_point_path, map_location='cuda:7')
    #
    src = torch.rand((1, 3, 224, 224))
    src = src.to(device)
    # out = model(src)
    out, mask1, mask2, mask3, mask4, mask5 = model(src)
    # out, mask1, mask2, mask3 = model(src)
    print(out)
    print(mask1)
    print(mask2)
    print(mask3)
    print(mask4)
    print(mask5)


    #
    # print(out.shape)
    # print(out[0, :10])
    # print('-' * 1000)

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


from config import get_args_parser
from models.all_elastic.mat_all_elastic import MatVisionTransformer
import torch
import torch.nn as nn
from functools import partial

def init(model, depth, init_width, target_width, check_point_path, args):
    model_state_dict = model.state_dict()

    checkpoint = torch.load(check_point_path, map_location=args.device)

    pretrained_state_dict = {k: v for k, v in checkpoint.items() if
                             k in model_state_dict and v.shape == model_state_dict[k].shape}
    model.load_state_dict(pretrained_state_dict, strict=False)

    with torch.no_grad():
        for i in range(depth):
            model.blocks[i].mlp.fc1.weight[:init_width, :] = checkpoint['blocks.' + str(i) + '.mlp.fc1.weight']
            model.blocks[i].mlp.fc1.bias[:init_width] = checkpoint['blocks.' + str(i) + '.mlp.fc1.bias']

            model.blocks[i].mlp.fc2.weight[:, :init_width] = checkpoint['blocks.' + str(i) + '.mlp.fc2.weight']
            model.blocks[i].mlp.fc2.bias[:] = checkpoint['blocks.' + str(i) + '.mlp.fc2.bias']

    return model

def init_cls_token(model, checkpoint, init_width):
    model.cls_token[:, :, :init_width] = checkpoint['cls_token']
    return model

def init_pos_embed(model, checkpoint, init_width):
    model.pos_embed[:, :, :init_width] = checkpoint['pos_embed']
    return model

def init_patch_embed(model, checkpoint, init_width):
    model.patch_embed.proj.weight[:init_width] = checkpoint['patch_embed.proj.weight']
    model.patch_embed.proj.bias[:init_width] = checkpoint['patch_embed.proj.bias']
    return model

def init_blocks(model, checkpoint, init_width, depth, width):
    model.blocks[depth].norm1.weight[:init_width] = checkpoint['blocks.' + str(depth) + '.norm1.weight']
    model.blocks[depth].norm1.bias[:init_width] = checkpoint['blocks.' + str(depth) + '.norm1.bias']

    model.blocks[depth].attn.qkv.weight[:init_width, :init_width] = checkpoint['blocks.' + str(depth) + '.attn.qkv.weight'][:init_width, :]
    model.blocks[depth].attn.qkv.weight[width:width+init_width, :init_width] = checkpoint['blocks.' + str(depth) + '.attn.qkv.weight'][init_width:init_width*2, :]
    model.blocks[depth].attn.qkv.weight[width*2:width*2+init_width, :init_width] = checkpoint['blocks.' + str(depth) + '.attn.qkv.weight'][init_width*2:init_width*3, :]

    model.blocks[depth].attn.qkv.bias[:init_width] = checkpoint['blocks.' + str(depth) + '.attn.qkv.bias'][:init_width]
    model.blocks[depth].attn.qkv.bias[width:width+init_width] = checkpoint['blocks.' + str(depth) + '.attn.qkv.bias'][init_width:init_width*2]
    model.blocks[depth].attn.qkv.bias[width*2:width*2+init_width] = checkpoint['blocks.' + str(depth) + '.attn.qkv.bias'][init_width*2:init_width*3]

    model.blocks[depth].attn.proj.weight[:init_width, :init_width] = checkpoint['blocks.' + str(depth) + '.attn.proj.weight']
    model.blocks[depth].attn.proj.bias[:init_width] = checkpoint['blocks.' + str(depth) + '.attn.proj.bias']

    model.blocks[depth].norm2.weight[:init_width] = checkpoint['blocks.' + str(depth) + '.norm2.weight']
    model.blocks[depth].norm2.bias[:init_width] = checkpoint['blocks.' + str(depth) + '.norm2.bias']

    model.blocks[depth].mlp.fc1.weight[:4*init_width, :init_width] = checkpoint['blocks.' + str(depth) + '.mlp.fc1.weight']
    model.blocks[depth].mlp.fc1.bias[:4*init_width] = checkpoint['blocks.' + str(depth) + '.mlp.fc1.bias']

    model.blocks[depth].mlp.fc2.weight[:init_width, :4*init_width] = checkpoint['blocks.' + str(depth) + '.mlp.fc2.weight']
    model.blocks[depth].mlp.fc2.bias[:init_width] = checkpoint['blocks.' + str(depth) + '.mlp.fc2.bias']

    return model

def init_norm_head(model, checkpoint, init_width):
    model.norm.weight[:init_width] = checkpoint['norm.weight']
    model.norm.bias[:init_width] = checkpoint['norm.bias']

    # model.head.weight[:, :init_width] = checkpoint['head.weight']
    # model.head.bias[:] = checkpoint['head.bias']

    return model

def init_v2(model, checkpoint, init_width, depth, width):
    with torch.no_grad():
        init_cls_token(model, checkpoint, init_width=init_width)
        init_pos_embed(model, checkpoint, init_width=init_width)
        init_patch_embed(model, checkpoint, init_width=init_width)
        for i in range(depth):
            init_blocks(model, checkpoint, init_width=init_width, depth=i, width=width)
        init_norm_head(model, checkpoint, init_width=init_width)

if __name__ == '__main__':
    args = get_args_parser()

    model = MatVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=100)

    model.to(args.device)
    model_state_dict = model.state_dict()

    check_point_path = '/home/ssd7T/zc_reuse/iccv/pretrained_para/vit_small.pth'
    checkpoint = torch.load(check_point_path, map_location=args.device)

    pretrained_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}

    model.load_state_dict(pretrained_state_dict, strict=False)

    init_v2(model, checkpoint, init_width=384, depth=12)

    print()

from config import get_args_parser
import torch
from tqdm import tqdm
from models.combined.combined_vit import MatVisionTransformer
from timm.loss import LabelSmoothingCrossEntropy
from utils.dataloader import build_cifar100_dataset_and_dataloader
from functools import partial
import torch.nn as nn
from ptflops import get_model_complexity_info
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
import re
import copy
import logging
from fvcore.nn import FlopCountAnalysis
from models.router.router_base import Router
from thop import profile
from inference_test import eval_mat, caculate_latency

if __name__ == '__main__':
    args = get_args_parser()
    valDataLoader = build_cifar100_dataset_and_dataloader(is_train=False, batch_size=args.batch_size, num_workers=args.num_workers, args=args)
    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    device = "cuda:5"

    model = MatVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes=100,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))

    model_path = '/home/ssd7T/zc_reuse/iccv/logs_weight/combined_stage_depth_300enone_cifar100/Feb08_21-14-21/weight/vit.pth'
    para = torch.load(model_path, map_location=device)
    model.load_state_dict(para, strict=False)
    model = model.to(device)
    model.eval()

    router = Router()
    router_path = '/home/ssd7T/zc_reuse/iccv/router_logs_weight/sm_router_together_lamda3none_cifar100/Feb12_12-46-00/weight/router.pth'
    para = torch.load(router_path, map_location=device)
    router.load_state_dict(para, strict=False)
    router = router.to(device)
    router.eval()

    input_tensor = torch.randn(1, 3, 224, 224).to(device=device)

    flag = None

    x = torch.ones((1, 1)).to(device)*0.7

    sub_dim, depth_list, mlp_ratio_list, mha_head_list, chosen_dim, probs = router(x)

    print(sub_dim)
    print(depth_list)
    print(mlp_ratio_list)
    print(mha_head_list)

    with torch.no_grad():
        model.configure_subnetwork(sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio_list, mha_head=mha_head_list)

    flops = FlopCountAnalysis(model, input_tensor)
    print(f"FLOPs: {flops.total()}")
    caculate_latency(model, flag, device=device)
    eval_mat(model, valDataLoader, criterion, device=device, sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio_list, mha_head=mha_head_list)



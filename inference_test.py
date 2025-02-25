from config import get_args_parser
import torch
from tqdm import tqdm
from models.final.model_easy_router import MatVisionTransformer, ModifiedBlock
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
from thop import profile

import matplotlib.pyplot as plt

def search_num(strings):
    match = re.search(r"[-+]?\d*\.\d+|\d+", strings)
    if match:
        number = float(match.group())
        if 'MMACs' in strings:
            number = number/1000
        # print(number)
    else:
        number = 0
        print("没有找到数字部分")
    return number

def eval_mat(model, valDataLoader, criterion, device, **kwargs):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description('eval')

        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(device)
                label = label.to(device)

                model.configure_subnetwork(**kwargs)
                preds = model(img)

                loss = criterion(preds, label)
                total_loss += loss.item()

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.update(1)

            avg_loss = total_loss / len(valDataLoader)
            accuracy = 100.0 * correct / total
            print("val loss", avg_loss)
            print("val acc", accuracy)

            pbar.close()
    return accuracy


def eval_dynamic(model, valDataLoader, device, latency):
    with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
        pbar.set_description('eval')

        model.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            for batch_idx, (img, label) in enumerate(valDataLoader):
                img = img.to(device)
                label = label.to(device)

                model.configure_latency(latency=latency)

                preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask, total_macs = model(img)

                _, predicted = torch.max(preds, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

                pbar.update(1)

            accuracy = 100.0 * correct / total
            print()
            print("val acc", accuracy)

            pbar.close()
    return accuracy


# def reset_model_hooks(model):
#     for module in model.modules():
#         module._backward_hooks = OrderedDict()
#         module._forward_hooks = OrderedDict()
#         module._forward_pre_hooks = OrderedDict()
def caculate_latency(model, flag, device):
    # model.configure_subnetwork(flag=flag)
    # with torch.cuda.device(device):
    #     flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    #     print(f"FLOPs: {flops}")
    #     print(f"Params: {params}")

    with get_accelerator().device(device):
        flops, macs, params = get_model_profile(model=model, # model
                                        input_shape=(1, 3, 224, 224), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                        args=None, # list of positional arguments to the model.
                                        kwargs=None, # dictionary of keyword arguments to the model.
                                        print_profile=True, # prints the model graph with the measured profile attached to each module
                                        detailed=True, # print the detailed profile
                                        module_depth=0, # depth into the nested modules, with -1 being the inner most modules
                                        top_modules=1, # the number of top modules to print aggregated profile
                                        warm_up=10, # the number of warm-ups before measuring the time of each module
                                        as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                        output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                        ignore_modules=None) # the list of modules to ignore in the profiling

        # print("flops", flops)
        print("macs", macs)
        # print("params", params)
    macs = search_num(macs)
    return macs


def get_flops(model, device):
    input_tensor = torch.randn(1, 3, 224, 224).to(device=device)
    flops = FlopCountAnalysis(model, input_tensor)
    return flops.total()/1e9


if __name__ == '__main__':
    args = get_args_parser()
    valDataLoader = build_cifar100_dataset_and_dataloader(is_train=False, batch_size=args.batch_size, num_workers=args.num_workers, args=args)
    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    device = "cuda:7"

    model = MatVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes=100,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block=ModifiedBlock)

    model_path = '/home-new/nus-zwb/reuse/code/train/train_final/logs_weight/random_tau1_hardfalse_lamda5_notacculate_adamw0.0011e-06_cifar100/Feb24_03-54-15/weight/dynamic_vit.pth'

    para = torch.load(model_path, map_location=device)
    model.load_state_dict(para, strict=False)

    model = model.to(device)
    model.eval()

    input_tensor = torch.randn(1, 3, 224, 224).to(device=device)

    flag = None

    latency = torch.tensor(0.6).to(device).unsqueeze(0)

    model.configure_latency(latency=latency)

    # mask_attn = torch.tensor([1, 1, 0]).to(device)
    # mask_mlp = torch.tensor([1, 1, 1]).to(device)
    # model.set_mask(mask_attn, mask_mlp)

    caculate_latency(model, flag, device=device)
    # flops = get_flops(model, device)
    # print(f"FLOPs:", flops)
    eval_dynamic(model, valDataLoader, device=device, latency=latency)


    #
    # sub_dim = 256
    # depth_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # mlp_ratio = 2
    # mha_head = 4
    # with torch.no_grad():
    #     model.configure_subnetwork(sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio, mha_head=mha_head)
    # # caculate_latency(model, flag, device=device)
    # flops = get_flops(model, device)
    # print(f"FLOPs:", flops)
    # eval_mat(model, valDataLoader, criterion, device=device, sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio, mha_head=mha_head)
    #

    # sub_dim_candidate = [256, 320, 384]
    # depth_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # depth_list_candidate = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    #
    # mlp_candidate = [1, 2, 4]
    # mha_candidate = [4, 5, 6]
    #
    # macs_list = []
    # acc_list = []
    #
    # for d in sub_dim_candidate:
    #     sub_dim = d
    #     for indice in depth_list_candidate:
    #         depth_list_tmp = [depth_list[i] for i in range(len(depth_list)) if i!=indice]
    #         mlp_ratio_list = []
    #         mha_heads_list = []
    #         for k in range(3):
    #             mlp_ratio_list = mlp_ratio_list + [mlp_candidate[k]]*4
    #             for m in range(3):
    #                 mlp_ratio_list = mlp_ratio_list + [mlp_candidate[m]] * 4
    #                 for n in range(3):
    #                     mlp_ratio_list = mlp_ratio_list + [mlp_candidate[n]] * 4
    #                     for q in range(3):
    #                         mha_heads_list = mha_heads_list+[mha_candidate[q]]*4
    #                         for w in range(3):
    #                             mha_heads_list = mha_heads_list + [mha_candidate[w]] * 4
    #                             for e in range(3):
    #                                 mha_heads_list = mha_heads_list + [mha_candidate[e]] * 4
    #
    #                                 print(sub_dim, depth_list_tmp, mlp_ratio_list, mha_heads_list)
    #
    #                                 with torch.no_grad():
    #                                     model.configure_subnetwork(sub_dim=sub_dim, depth_list=depth_list_tmp,
    #                                                                mlp_ratio=mlp_ratio_list, mha_head=mha_heads_list)
    #                                 flops = get_flops(model, device)
    #                                 print(f"FLOPs:", flops)
    #                                 acc = eval_mat(model, valDataLoader, criterion, device=device, sub_dim=sub_dim,
    #                                                depth_list=depth_list_tmp, mlp_ratio=mlp_ratio_list, mha_head=mha_heads_list)
    #                                 macs_list.append(flops)
    #                                 acc_list.append(acc)
    #                                 mha_heads_list = mha_heads_list[:-4]
    #                             mha_heads_list = mha_heads_list[:-4]
    #                         mha_heads_list = mha_heads_list[:-4]
    #                     mlp_ratio_list = mlp_ratio_list[:-4]
    #                 mlp_ratio_list = mlp_ratio_list[:-4]
    #             mlp_ratio_list = mlp_ratio_list[:-4]
    #
    # plt.scatter(macs_list, acc_list)
    # plt.show()
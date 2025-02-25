from config import get_args_parser
from datetime import datetime
import os
import torch
from tqdm import tqdm
from models.combined.combined_vit import MatVisionTransformer
from utils.set_wandb import set_wandb
import wandb
from timm.loss import LabelSmoothingCrossEntropy
from utils.dataloader import build_cifar100_dataset_and_dataloader, bulid_dataloader
from utils.lr_sched import adjust_learning_rate
from utils.eval_flag import eval_mat, eval_mat_fined, eval_mat_combined
import random
from models.router.router_base import Router
from models.router.router_sm import SM
from models.router.router_base import Router
from utils.initial import init, init_v2
from inference_test import caculate_latency, get_flops
import torch.nn.functional as F


flags_list = ['s', 'm', 'l']

def train(args):
    torch.cuda.set_device(args.device)
    model = MatVisionTransformer(embed_dim=args.initial_embed_dim, depth=args.initial_depth,
                               num_heads=args.initial_embed_dim//64, num_classes=args.nb_classes,
                               drop_path_rate=args.drop_path, mlp_ratio=args.mlp_ratio, qkv_bias=True)
    model.to(args.device)

    model.eval()

    check_point_path = '/home/ssd7T/zc_reuse/iccv/logs_weight/combined_stage_depth_300enone_cifar100/Feb08_21-14-21/weight/vit.pth'
    checkpoint = torch.load(check_point_path, map_location=args.device)
    model.load_state_dict(checkpoint, strict=False)

    sm = SM()
    sm.to(args.device)
    check_point_path = '/home/ssd7T/zc_reuse/iccv/router_logs_weight/smnone_cifar100/Feb11_20-25-30/weight/sm.pth'
    checkpoint = torch.load(check_point_path, map_location=args.device)
    sm.load_state_dict(checkpoint, strict=False)

    router = Router()
    router.to(args.device)
    check_point_path = '/home/ssd7T/zc_reuse/iccv/router_logs_weight/router_w_smnone_cifar100/Feb11_19-40-44/weight/vit.pth'
    checkpoint = torch.load(check_point_path, map_location=args.device)
    router.load_state_dict(checkpoint, strict=False)

    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    trainDataLoader = build_cifar100_dataset_and_dataloader(is_train=True, batch_size=args.batch_size, num_workers=args.num_workers,args=args)
    valDataLoader = build_cifar100_dataset_and_dataloader(is_train=False, batch_size=args.batch_size, num_workers=args.num_workers, args=args)

    optimizer_router = torch.optim.AdamW(router.parameters(), lr=args.lr)
    optimizer_sm = torch.optim.AdamW(sm.parameters(), lr=args.lr)


    folder_path = 'router_logs_weight/'+args.model+args.expand_method+args.dataset

    os.makedirs(folder_path, exist_ok=True)
    time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(folder_path, time)
    os.makedirs(log_dir)

    weight_path = os.path.join(log_dir, 'weight')
    os.makedirs(weight_path)

    set_wandb(args)

    accmulated_step = 16

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_router, epoch + 1, args)
        adjust_learning_rate(optimizer_sm, epoch + 1, args)

        router.train()
        sm.eval()
        router_total_loss = 0
        sm_total_loss = 0

        for i in range(accmulated_step):
            wandb.log({"Epoch": epoch + 1, "router_learning_rate": optimizer_router.param_groups[0]['lr']})

            x = torch.rand((1, 1)).to(args.device)

            sub_dim, depth_list, mlp_ratio_list, mha_head_list, chosen_dim, probs= router(x)

            sub_dim, depth_list, mlp_ratio_list, mha_head_list, preds = sm(probs)

            pred_loss = preds[0][0]
            pred_macs = preds[0][1]

            router_loss = 3*pred_loss+x*pred_macs
            wandb.log({"router_loss": router_loss.item()})
            router_total_loss += router_loss.item()
            router_loss.backward()

        wandb.log({"Epoch": epoch + 1, "router_total_loss": router_total_loss / accmulated_step})

        optimizer_router.step()
        optimizer_router.zero_grad()


        router.eval()
        sm.train()

        for i in range(accmulated_step):
            wandb.log({"Epoch": epoch + 1, "sm_learning_rate": optimizer_sm.param_groups[0]['lr']})
            x = torch.rand((1, 1)).to(args.device)

            sub_dim, depth_list, mlp_ratio_list, mha_head_list, chosen_dim, probs= router(x)

            sub_dim, depth_list, mlp_ratio_list, mha_head_list, preds = sm(probs)

            pred_loss = preds[0][0]
            pred_macs = preds[0][1]

            with torch.no_grad():
                for batch_idx, (img, label) in enumerate(valDataLoader):
                    img = img.to(args.device)
                    label = label.to(args.device)

                    model.configure_subnetwork(sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio_list,
                                               mha_head=mha_head_list)
                    macs = get_flops(model, device=args.device)
                    macs = torch.tensor(macs).to(args.device)
                    model_preds = model(img)
                    gt_train_loss = criterion(model_preds, label)
                    break

            sm_loss = F.mse_loss(input=pred_loss, target=gt_train_loss)+F.mse_loss(input=pred_macs, target=macs)

            wandb.log({"gt_train_loss": gt_train_loss.item()})
            wandb.log({"sm_pred_train_loss": pred_loss.item()})
            wandb.log({"gt_macs": macs})
            wandb.log({"sm_pred_macs": pred_macs.item()})

            wandb.log({"sm_loss": sm_loss.item()})
            sm_total_loss += sm_loss.item()
            sm_loss.backward()

        wandb.log({"Epoch": epoch + 1, "sm_total_loss": sm_total_loss/accmulated_step})

        optimizer_sm.step()
        optimizer_sm.zero_grad()

        with torch.no_grad():
            router.eval()
            for index, f in enumerate(flags_list):
                x = torch.tensor([[index*0.4]]).to(args.device)

                sub_dim, depth_list, mlp_ratio_list, mha_head_list, chosen_dim, probs = router(x)

                eval_mat_combined(model, valDataLoader, criterion, epoch, optimizer_router, args, flag=f, sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio_list,
                                           mha_head=mha_head_list)

        torch.save(sm.state_dict(), weight_path+'/sm.pth')
        torch.save(router.state_dict(), weight_path + '/router.pth')


if __name__ == '__main__':
    args = get_args_parser()
    train(args)





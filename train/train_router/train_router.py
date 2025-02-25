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
from utils.initial import init, init_v2
from inference_test import caculate_latency

flags_list = ['s', 'm', 'l']

def train(args):
    torch.cuda.set_device(args.device)
    model = MatVisionTransformer(embed_dim=args.initial_embed_dim, depth=args.initial_depth,
                               num_heads=args.initial_embed_dim//64, num_classes=args.nb_classes,
                               drop_path_rate=args.drop_path, mlp_ratio=args.mlp_ratio, qkv_bias=True)
    model.to(args.device)
    router = Router()
    router.to(args.device)

    if args.pretrained:
        check_point_path = '/home/ssd7T/zc_reuse/iccv/logs_weight/combined_stage_depth_300enone_cifar100/Feb08_21-14-21/weight/vit.pth'
        checkpoint = torch.load(check_point_path, map_location=args.device)
        model.load_state_dict(checkpoint, strict=False)
        # init_v2(model, checkpoint, init_width=384, depth=12, width=384)
    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model.eval()


    # trainDataLoader = bulid_dataloader(is_train=True, args=args)
    # valDataLoader = bulid_dataloader(is_train=False, args=args)

    trainDataLoader = build_cifar100_dataset_and_dataloader(is_train=True, batch_size=args.batch_size, num_workers=args.num_workers,args=args)
    valDataLoader = build_cifar100_dataset_and_dataloader(is_train=False, batch_size=args.batch_size, num_workers=args.num_workers, args=args)

    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr)

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
        adjust_learning_rate(optimizer, epoch + 1, args)
        router.train()
        total_loss = 0
        for i in range(accmulated_step):
            wandb.log({"Epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]['lr']})

            # x = torch.rand((1, 1)).to(args.device)
            x = torch.ones((1, 1)).to(args.device)

            sub_dim, depth_list, mlp_ratio_list, mha_head_list, chosen_dim = router(x)

            with torch.no_grad():
                for batch_idx, (img, label) in enumerate(trainDataLoader):
                    img = img.to(args.device)
                    label = label.to(args.device)

                    model.configure_subnetwork(sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio_list,
                                               mha_head=mha_head_list)
                    # macs = caculate_latency(model, flag=None, device=args.device)
                    preds = model(img)
                    train_loss = criterion(preds, label)
                    # reward = -train_loss-macs*x
                    reward = -train_loss

                    wandb.log({"model_train_loss": train_loss.item()})
                    # wandb.log({"macs": macs})
                    # wandb.log({"macs*x": (macs*x).item()})
                    wandb.log({"reward": reward.item()})

                    break

            log_prob = torch.log(torch.mean(chosen_dim))
            wandb.log({"log_prob": log_prob.item()})

            # print(log_prob)
            # print(reward)

            loss = -log_prob*reward
            wandb.log({"loss": loss.item()})
            total_loss += loss.item()
            loss.backward()

        wandb.log({"Epoch": epoch + 1, "total_loss": total_loss/accmulated_step})

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            router.eval()
            for index, f in enumerate(flags_list):
                # x = torch.tensor([[index*0.4]]).to(args.device)
                x = torch.ones((1, 1)).to(args.device)
                sub_dim, depth_list, mlp_ratio_list, mha_head_list, chosen_dim = router(x)

                eval_mat_combined(model, valDataLoader, criterion, epoch, optimizer, args, flag=f, sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio_list,
                                           mha_head=mha_head_list)

        torch.save(router.state_dict(), weight_path+'/vit.pth')


if __name__ == '__main__':
    args = get_args_parser()
    train(args)





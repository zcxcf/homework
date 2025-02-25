import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from config import get_args_parser
from datetime import datetime
import os
import torch
import timm
from tqdm import tqdm
from utils.set_wandb import set_wandb
import wandb
from models.final.model_base import MatVisionTransformer, ModifiedBlock
from timm.loss import LabelSmoothingCrossEntropy
from utils.dataloader import build_cifar100_dataset_and_dataloader
from utils.lr_sched import adjust_learning_rate
from utils.eval_flag import eval_mat, eval_mat_fined, eval_mat_combined, eval_mat_dynamic, eval_mask, eval_all
from utils.initial import init, init_v2
import random

flags_list = ['0.2', '0.4', '0.6', '0.8', '1.0']

latency_list = [0.2, 0.4, 0.6, 0.8, 1.0]

def train(args):
    torch.cuda.set_device(args.device)
    model = MatVisionTransformer(embed_dim=args.initial_embed_dim, depth=args.initial_depth,
                                 num_heads=args.initial_embed_dim // 64, num_classes=args.nb_classes,
                                 drop_path_rate=args.drop_path, mlp_ratio=args.mlp_ratio, qkv_bias=True, block=ModifiedBlock)
    model.to(args.device)
    # model_path = '/home-new/nus-zwb/reuse/code/train/train_mat/logs_weight/mat_smallnone_cifar100/Feb19_05-43-29/weight/vit.pth'
    model_path = '/home-new/nus-zwb/reuse/code/pretrained_para/vit_small.pth'
    para = torch.load(model_path, map_location=args.device)
    model.load_state_dict(para, strict=False)
    # check_point_path = '/home/ssd7T/zc_reuse/iccv/pretrained_para/vit_small.pth'
    # checkpoint = torch.load(check_point_path, map_location=args.device)
    # init_v2(model, checkpoint, init_width=384, depth=12, width=384)

    for name, param in model.named_parameters():
        param.requires_grad = True
        # if 'router' not in name:
        #     param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable")
        else:
            print(f"{name} is frozen")

    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # trainDataLoader = bulid_dataloader(is_train=True, args=args)
    # valDataLoader = bulid_dataloader(is_train=False, args=args)

    trainDataLoader = build_cifar100_dataset_and_dataloader(is_train=True, batch_size=args.batch_size, num_workers=args.num_workers, args=args)
    valDataLoader = build_cifar100_dataset_and_dataloader(is_train=False, batch_size=args.batch_size, num_workers=args.num_workers, args=args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    folder_path = 'logs_weight/' + args.model + args.expand_method + args.dataset

    os.makedirs(folder_path, exist_ok=True)
    time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(folder_path, time)
    os.makedirs(log_dir)

    weight_path = os.path.join(log_dir, 'weight')
    os.makedirs(weight_path)

    set_wandb(args)

    completed_steps = 0

    for epoch in range(args.epochs):
        with tqdm(total=len(trainDataLoader), postfix=dict, mininterval=0.3) as pbar:
            pbar.set_description(f'train Epoch {epoch + 1}/{args.epochs}')

            adjust_learning_rate(optimizer, epoch + 1, args)

            wandb.log({"Epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]['lr']})

            model.train()

            total_loss = 0
            total_ce_loss = 0
            total_latency_loss = 0

            total_attn_mask = 0
            total_mlp_mask = 0
            total_embed_mask = 0
            total_depth_mlp_mask = 0
            total_depth_attn_mask = 0

            for batch_idx, (img, label) in enumerate(trainDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)
                optimizer.zero_grad()

                # r = random.randint(0, 4)
                # latency = torch.tensor(latency_list[r]).to(args.device).unsqueeze(0)
                # print(latency)
                #
                latency = torch.rand(1).to(args.device)

                model.configure_latency(latency=latency)

                preds, attn_mask, mlp_mask, embed_mask, depth_attn_mask, depth_mlp_mask = model(img)

                # preds, cost = model(img)

                ce_loss = criterion(preds, label)

                latency_loss = torch.square(latency-attn_mask) + torch.square(latency-mlp_mask) + torch.square(latency-embed_mask) + torch.square(latency-depth_mlp_mask) + torch.square(latency-depth_attn_mask)

                # if mask < latency:
                #     latency_loss = torch.tensor(0)
                # else:
                #     latency_loss = torch.square(latency - mask)

                # latency_loss = (1-latency) * cost*0.5

                loss = ce_loss + latency_loss * 5

                if batch_idx % 10 == 0:
                    wandb.log({"batch cross entropy loss": ce_loss})
                    wandb.log({"batch latency loss": latency_loss})
                    wandb.log({"train Batch Loss": loss.item()})

                    wandb.log({"train Batch attn mask": attn_mask.item()})
                    wandb.log({"train Batch mlp mask": mlp_mask.item()})
                    wandb.log({"train Batch embed mask": embed_mask.item()})

                    wandb.log({"train Batch depth mlp mask": depth_mlp_mask.item()})
                    wandb.log({"train Batch depth attn mask": depth_attn_mask.item()})


                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_latency_loss += latency_loss.item()
                total_attn_mask += attn_mask.item()
                total_mlp_mask += mlp_mask.item()
                total_embed_mask += embed_mask.item()
                total_depth_mlp_mask += depth_mlp_mask.item()
                total_depth_attn_mask += depth_attn_mask.item()


                loss.backward()
                optimizer.step()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

                completed_steps += 1

            epoch_loss = total_loss / len(trainDataLoader)
            epoch_ce_loss = total_ce_loss / len(trainDataLoader)
            epoch_latency_loss = total_latency_loss / len(trainDataLoader)

            epoch_attn_mask = total_attn_mask / len(trainDataLoader)
            epoch_mlp_mask = total_mlp_mask / len(trainDataLoader)
            epoch_embed_mask = total_embed_mask / len(trainDataLoader)
            epoch_depth_mlp_mask = total_depth_mlp_mask / len(trainDataLoader)
            epoch_depth_attn_mask = total_depth_attn_mask / len(trainDataLoader)


            print("train loss", epoch_loss)

            wandb.log({"Epoch": epoch + 1, "Train epoch Loss": epoch_loss})
            wandb.log({"Epoch": epoch + 1, "Train epoch cross entropy loss": epoch_ce_loss})
            wandb.log({"Epoch": epoch + 1, "Train epoch latency loss": epoch_latency_loss})
            wandb.log({"Epoch": epoch + 1, "Train attn mask": epoch_attn_mask})
            wandb.log({"Epoch": epoch + 1, "Train mlp mask": epoch_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "Train embed mask": epoch_embed_mask})
            wandb.log({"Epoch": epoch + 1, "Train depth mlp mask": epoch_depth_mlp_mask})
            wandb.log({"Epoch": epoch + 1, "Train depth attn mask": epoch_depth_attn_mask})

            pbar.close()

        if epoch % 2 == 0:

            for index, f in enumerate(flags_list):
                latency = torch.tensor(latency_list[index]).to(args.device).unsqueeze(0)
                eval_all(model, valDataLoader, criterion, epoch, optimizer, args, flag=f, latency=latency)

        torch.save(model.state_dict(), weight_path + '/dynamic_vit.pth')


if __name__ == '__main__':
    args = get_args_parser()
    train(args)





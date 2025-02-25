import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

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
from utils.initial import init, init_v2

flags_list = ['l', 'm', 's', 'ss', 'sss']

mlp_ratio_list = [4, 3, 2, 1, 0.5]

mha_head_list = [12, 11, 10, 9, 8]

eval_mlp_ratio_list = [4, 3, 2, 1, 0.5]

eval_mha_head_list = [12, 11, 10, 9, 8]


def train(args):
    torch.cuda.set_device(args.device)
    model = MatVisionTransformer(embed_dim=args.initial_embed_dim, depth=args.initial_depth,
                               num_heads=args.initial_embed_dim//64, num_classes=args.nb_classes,
                               drop_path_rate=args.drop_path, mlp_ratio=args.mlp_ratio, qkv_bias=True)
    model.to(args.device)

    if args.pretrained:
        check_point_path = '/home-new/nus-zwb/reuse/code/pretrained_para/vit_base.pth'
        # model = init(model, depth=args.initial_depth, init_width=768, target_width=3072,
        #              check_point_path=check_point_path, args=args)
        checkpoint = torch.load(check_point_path, map_location=args.device)
        init_v2(model, checkpoint, init_width=768, depth=12, width=768)

    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # trainDataLoader = bulid_dataloader(is_train=True, args=args)
    # valDataLoader = bulid_dataloader(is_train=False, args=args)

    trainDataLoader = build_cifar100_dataset_and_dataloader(is_train=True, batch_size=args.batch_size, num_workers=args.num_workers,args=args)
    valDataLoader = build_cifar100_dataset_and_dataloader(is_train=False, batch_size=args.batch_size, num_workers=args.num_workers, args=args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    folder_path = 'logs_weight/'+args.model+args.dataset+str(args.lr)

    os.makedirs(folder_path, exist_ok=True)
    time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(folder_path, time)
    os.makedirs(log_dir)

    weight_path = os.path.join(log_dir, 'weight')
    os.makedirs(weight_path)

    set_wandb(args)

    current_stage = 0

    for epoch in range(args.epochs):

        with tqdm(total=len(trainDataLoader), postfix=dict, mininterval=0.3) as pbar:
            pbar.set_description(f'train Epoch {epoch + 1}/{args.epochs}')

            adjust_learning_rate(optimizer, epoch+1, args)

            wandb.log({"Epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]['lr']})

            model.train()
            total_loss = 0

            if epoch in args.stage_epochs:
                stage_index = args.stage_epochs.index(epoch)
                wandb.log({"Epoch": epoch + 1, "stage": stage_index})
                current_stage += 1

            for batch_idx, (img, label) in enumerate(trainDataLoader):

                img = img.to(args.device)
                label = label.to(args.device)
                optimizer.zero_grad()

                loss = 0

                r = random.randint(0, current_stage)

                sub_dim = 64*mha_head_list[r]

                # r = random.randint(0, current_stage)

                mha_head = mha_head_list[r]
                #
                # sub_dim = 768
                # mha_head = 12

                depth_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                r = random.randint(0, 5)
                if r > 2:
                    r = 0
                if r>0:
                    num_to_remove = random.choice(list(range(r)))
                    indices_to_remove = random.sample(range(len(depth_list)), num_to_remove)
                    depth_list = [depth_list[i] for i in range(len(depth_list)) if i not in indices_to_remove]

                r = random.randint(0, current_stage)

                mlp_ratio = mlp_ratio_list[r]

                # mlp_ratio = 4

                model.configure_subnetwork(sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio,
                                           mha_head=mha_head)

                preds = model(img)
                loss += criterion(preds, label)

                if batch_idx % 10 == 0:
                    wandb.log({"train Batch Loss": loss.item()})
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            epoch_loss = total_loss / len(trainDataLoader)
            print("train loss", epoch_loss)
            wandb.log({"Epoch": epoch + 1, "Train epoch Loss": epoch_loss})

            pbar.close()

        if epoch % 2 == 0:
            for index, f in enumerate(flags_list):
                sub_dim = 64 * eval_mha_head_list[index]
                mha_head = eval_mha_head_list[index]
                # sub_dim = 768
                # mha_head = 12
                depth_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                mlp_ratio = eval_mlp_ratio_list[index]
                # mlp_ratio = 4
                eval_mat_combined(model, valDataLoader, criterion, epoch, optimizer, args, flag=f, sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio,
                                           mha_head=mha_head)

        torch.save(model.state_dict(), weight_path+'/matformer.pth')


if __name__ == '__main__':
    args = get_args_parser()
    train(args)





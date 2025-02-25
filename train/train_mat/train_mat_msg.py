from config import get_args_parser
from datetime import datetime
import os
import torch
from tqdm import tqdm
from models.all_elastic.msg import MatVisionTransformer
from utils.set_wandb import set_wandb
import wandb
from timm.loss import LabelSmoothingCrossEntropy
from utils.dataloader import build_cifar100_dataset_and_dataloader, bulid_dataloader
from utils.lr_sched import adjust_learning_rate
from utils.eval_flag import eval_mat
import random
from utils.initial import init, init_v2

flags_list = ['s', 'm', 'l']

def train(args):
    torch.cuda.set_device(args.device)
    model = MatVisionTransformer(embed_dim=args.initial_embed_dim, depth=args.initial_depth,
                               num_heads=args.initial_embed_dim//64, num_classes=args.nb_classes,
                               drop_path_rate=args.drop_path, mlp_ratio=args.mlp_ratio, qkv_bias=True)
    model.to(args.device)
    if args.pretrained:
        check_point_path = '/home/ssd7T/zc_reuse/iccv/pretrained_para/vit_tiny.pth'
        # model = init(model, depth=args.initial_depth, init_width=768, target_width=3072,
        #              check_point_path=check_point_path, args=args)
        checkpoint = torch.load(check_point_path, map_location=args.device)
        init_v2(model, checkpoint, init_width=192, depth=12)

    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    trainDataLoader = bulid_dataloader(is_train=True, args=args)
    valDataLoader = bulid_dataloader(is_train=False, args=args)

    # trainDataLoader = build_cifar100_dataset_and_dataloader(is_train=True, batch_size=args.batch_size, num_workers=args.num_workers,args=args)
    # valDataLoader = build_cifar100_dataset_and_dataloader(is_train=False, batch_size=args.batch_size, num_workers=args.num_workers, args=args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    folder_path = 'logs_weight/'+args.model+args.expand_method+args.dataset

    os.makedirs(folder_path, exist_ok=True)
    time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(folder_path, time)
    os.makedirs(log_dir)

    weight_path = os.path.join(log_dir, 'weight')
    os.makedirs(weight_path)

    set_wandb(args)

    completed_steps = 0
    current_stage = 0

    for epoch in range(args.epochs):

        with tqdm(total=len(trainDataLoader), postfix=dict, mininterval=0.3) as pbar:
            pbar.set_description(f'train Epoch {epoch + 1}/{args.epochs}')

            adjust_learning_rate(optimizer, epoch+1, args)

            wandb.log({"Epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]['lr']})

            total_loss = 0

            model.eval()
            with torch.no_grad():
                model.configure_subnetwork(flag=flags_list[current_stage])
                if epoch in args.stage_epochs:
                    stage_index = args.stage_epochs.index(epoch)
                    wandb.log({"Epoch": epoch + 1, "stage": stage_index})
                    # model.expand_subnetwork(type='fpi_pre_small')
                    current_stage += 1

            for batch_idx, (img, label) in enumerate(trainDataLoader):
                model.eval()
                model.configure_subnetwork(flag=current_stage)
                with torch.no_grad():
                    model.updata_mask(step=1/4000)

                model.train()

                img = img.to(args.device)
                label = label.to(args.device)
                optimizer.zero_grad()

                loss = 0

                r = random.randint(0, current_stage)

                f_tmp = flags_list[r]
                model.configure_subnetwork(flag=f_tmp)
                preds = model(img)
                loss += criterion(preds, label)

                if batch_idx % 10 == 0:
                    wandb.log({"train Batch Loss": loss.item()})
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

                completed_steps += 1

            epoch_loss = total_loss / len(trainDataLoader)
            print("train loss", epoch_loss)
            wandb.log({"Epoch": epoch + 1, "Train epoch Loss": epoch_loss})

            pbar.close()

        if epoch % 2 == 0:
            for f in flags_list:
                eval_mat(model, valDataLoader, criterion, epoch, optimizer, args, flag=f)

        torch.save(model.state_dict(), weight_path+'/vit.pth')


if __name__ == '__main__':
    args = get_args_parser()
    train(args)





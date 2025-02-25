import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from config import get_args_parser
from datetime import datetime
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from timm.data.mixup import Mixup
from models.combined.combined_vit import MatVisionTransformer
from utils.set_wandb import set_wandb
import wandb
from torch.utils.data.distributed import DistributedSampler
from utils.dataloader import build_cifar100_dataset_and_dataloader, bulid_dataloader, build_dataset
from utils.lr_sched import adjust_learning_rate
from utils.eval_flag import eval_mat, eval_mat_fined, eval_mat_combined, eval_mat_combined_dis
import random
from utils.initial import init, init_v2
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

flags_list = ['l', 'm', 's', 'ss', 'sss']

mlp_ratio_list = [4, 3, 2, 1, 0.5]

mha_head_list = [12, 11, 10, 9, 8]

eval_mlp_ratio_list = [4, 3, 2, 1, 0.5]

eval_mha_head_list = [12, 11, 10, 9, 8]


def train(args):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.distributed.init_process_group(backend='nccl')

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = MatVisionTransformer(embed_dim=args.initial_embed_dim, depth=args.initial_depth,
                               num_heads=args.initial_embed_dim//64, num_classes=args.nb_classes,
                               drop_path_rate=args.drop_path, mlp_ratio=args.mlp_ratio, qkv_bias=True)
    model.to(device)

    if args.pretrained:
        # check_point_path = '/home-new/nus-zwb/reuse/code/pretrained_para/vit_base.pth'
        check_point_path = '/home/ssd7T/zc_reuse/iccv/pretrained_para/vit_base.pth'
        checkpoint = torch.load(check_point_path, map_location=device)
        init_v2(model, checkpoint, init_width=768, depth=12, width=768)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    train_dataset = build_dataset(is_train=True, args=args)
    val_dataset = build_dataset(is_train=False, args=args)

    trainDataLoader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=DistributedSampler(train_dataset))
    valDataLoader = DataLoader(dataset=val_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if torch.distributed.get_rank() == 0:
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

            if torch.distributed.get_rank() == 0:
                wandb.log({"Epoch": epoch + 1, "learning_rate" + str(local_rank): optimizer.param_groups[0]['lr']})

            model.train()
            total_loss = 0

            if epoch in args.stage_epochs:
                stage_index = args.stage_epochs.index(epoch)
                if torch.distributed.get_rank() == 0:
                    wandb.log({"Epoch": epoch + 1, "stage" + str(local_rank): stage_index})
                current_stage += 1

            for batch_idx, (img, label) in enumerate(trainDataLoader):

                img = img.to(device)
                label = label.to(device)

                if mixup_fn is not None:
                    img, label = mixup_fn(img, label)

                optimizer.zero_grad()

                loss = 0

                r = random.randint(0, current_stage)

                sub_dim = 64*mha_head_list[r]

                # r = random.randint(0, current_stage)

                mha_head = mha_head_list[r]

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

                model.module.configure_subnetwork(sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio,
                                           mha_head=mha_head)

                preds = model(img)
                loss += criterion(preds, label)

                if torch.distributed.get_rank() == 0:
                    if batch_idx % 10 == 0:
                        wandb.log({"train Batch Loss" + str(local_rank): loss.item()})

                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                pbar.update(1)

            epoch_loss = total_loss / len(trainDataLoader)
            print("train loss", epoch_loss)
            if torch.distributed.get_rank() == 0:
                wandb.log({"Epoch": epoch + 1, "Train epoch Loss" + str(local_rank): epoch_loss})

            pbar.close()

        if torch.distributed.get_rank() == 0:
            if epoch % 2 == 0:
                for index, f in enumerate(flags_list):
                    sub_dim = 64 * eval_mha_head_list[index]
                    mha_head = eval_mha_head_list[index]
                    # sub_dim = 768
                    # mha_head = 12
                    depth_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                    mlp_ratio = eval_mlp_ratio_list[index]
                    # mlp_ratio = 4
                    eval_mat_combined_dis(model, valDataLoader, criterion, epoch, optimizer, args, flag=f, sub_dim=sub_dim, depth_list=depth_list, mlp_ratio=mlp_ratio,
                                               mha_head=mha_head, device=device, local_rank=local_rank)

        if torch.distributed.get_rank() == 0:
            torch.save(model.state_dict(), weight_path+'/matformer.pth')

if __name__ == '__main__':
    # torch.distributed.init_process_group(backend="nccl")
    args = get_args_parser()
    train(args)





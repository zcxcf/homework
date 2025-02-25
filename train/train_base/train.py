from config import get_args_parser
from datetime import datetime
import os
import torch
from tqdm import tqdm
from models.history import model_vits, methods
from utils.set_wandb import set_wandb
import wandb
from timm.loss import LabelSmoothingCrossEntropy
from utils.dataloader import build_cifar100_dataset_and_dataloader
from utils.lr_sched import adjust_learning_rate


def train(args):
    model = model_vits.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    model.to(args.device)

    if args.initial !='_from_scratch':
        checkpoint = torch.load(args.initial_path, map_location=args.device)
        if 'mae' in args.initial:
            checkpoint_model = checkpoint['model']
        elif 'from_tiny' in args.initial:
            checkpoint_model = methods.__dict__[args.expand_method](checkpoint, 192, 576, 12, 16, [2, 5, 8, 11])
        else:
            checkpoint_model = checkpoint

        model_dict = model.state_dict()

        if args.initial_head:
            pretrained_dict = {key: value for key, value in checkpoint_model.items() if
                               (key in model_dict)}
        else:
            pretrained_dict = {key: value for key, value in checkpoint_model.items() if
                                (key in model_dict and 'head' not in key)}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # trainDataLoader = bulid_dataloader(is_train=True, args=args)
    # valDataLoader = bulid_dataloader(is_train=False, args=args)

    trainDataLoader = build_cifar100_dataset_and_dataloader(is_train=True, batch_size=16, num_workers=4)
    valDataLoader = build_cifar100_dataset_and_dataloader(is_train=False, batch_size=16, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    folder_path = 'logs_weight/'+args.model+args.initial+args.expand_method+args.dataset

    os.makedirs(folder_path, exist_ok=True)
    time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(folder_path, time)
    os.makedirs(log_dir)

    weight_path = os.path.join(log_dir, 'weight')
    os.makedirs(weight_path)

    set_wandb(args)

    for epoch in range(args.epochs):

        with tqdm(total=len(trainDataLoader), postfix=dict, mininterval=0.3) as pbar:
            pbar.set_description(f'train Epoch {epoch + 1}/{args.epochs}')

            adjust_learning_rate(optimizer, epoch+1, args)

            wandb.log({"Epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]['lr']})

            model.train()
            total_loss = 0
            for batch_idx, (img, label) in enumerate(trainDataLoader):
                img = img.to(args.device)
                label = label.to(args.device)

                preds = model(img)

                optimizer.zero_grad()

                loss = criterion(preds, label)

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

        if epoch % 5 == 0:

            with tqdm(total=len(valDataLoader), postfix=dict, mininterval=0.3) as pbar:
                pbar.set_description(f'eval Epoch {epoch + 1}/{args.epochs}')

                model.eval()

                with torch.no_grad():
                    total_loss = 0.0
                    correct = 0
                    total = 0
                    for batch_idx, (img, label) in enumerate(valDataLoader):
                        img = img.to(args.device)
                        label = label.to(args.device)
                        preds = model(img)

                        loss = criterion(preds, label)
                        total_loss += loss.item()

                        _, predicted = torch.max(preds, 1)
                        total += label.size(0)
                        correct += (predicted == label).sum().item()

                        pbar.set_postfix(**{"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                        pbar.update(1)

                    avg_loss = total_loss / len(valDataLoader)
                    accuracy = 100.0 * correct / total
                    print("val loss", avg_loss)
                    print("val acc", accuracy)
                    wandb.log({"Epoch": epoch + 1, "Val Loss": avg_loss})
                    wandb.log({"Epoch": epoch + 1, "Val Acc": accuracy})

                    pbar.close()

        torch.save(model.state_dict(), weight_path+'/vit.pth')


if __name__ == '__main__':
    args = get_args_parser()
    train(args)





import wandb


def set_wandb(args):
    wandb.init(
        project="iccv_imagenet",
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        name=args.model+args.dataset+'_lr'+str(args.lr)+str(args.min_lr)+'_epoch'+str(args.epochs)
    )


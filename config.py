import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('reuse', add_help=False)

    # train
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)

    parser.add_argument('--local_rank', default=-1, type=int)

    # Model parameters
    parser.add_argument('--model', default='imagenet_15_20_25_30', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--initial_embed_dim', default=768, type=int)
    parser.add_argument('--initial_depth', default=12, type=int)
    parser.add_argument('--mlp_ratio', default=4, type=int)

    parser.add_argument('--pretrained', default=True, type=bool)

    parser.add_argument('--expand_method', default='none', type=str, metavar='MODEL', help='Name of expand method')

    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

    parser.add_argument('--global_pool', default='avg', type=str)

    # grow parameters
    parser.add_argument('--stage_steps', default=[7000, 14000], type=list)

    parser.add_argument('--stage_epochs', default=[15, 20, 25, 30], type=list)

    parser.add_argument('--stage_fined_epochs', default=[10, 12, 14, 16, 18, 20, 22, 24], type=list)

    parser.add_argument('--grow_steps', default=[25000, 50000, 75000], type=list)

    parser.add_argument('--grow_embed_dims', default=[288, 384, 576], type=list)
    parser.add_argument('--grow_depth', default=[8, 10, 12], type=list)
    parser.add_argument('--grow_depth_insert_position', default=[[2, 5], [3, 7], [4, 9]], type=list)


    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=5e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=0.8,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters

    # parser.add_argument('--data_path', default=r'E:\imagenet100', type=str,
    #                     help='dataset path')
    # parser.add_argument('--data_path', default=r'/home/ssd7T/zc_reuse/imagenet100tiny', type=str,
    #                     help='dataset path')

    parser.add_argument('--data_path', default=r'/home/wangkai/big_space/datasets/imagenet', type=str,
                        help='dataset path')

    # parser.add_argument('--data_path', default=r'/data/common/ImageNet', type=str,
    #                     help='dataset path')

    parser.add_argument('--dataset', default='_imagenet', type=str,
                        help='dataset name')

    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')

    parser.add_argument('--device', default='cuda:4',
                        help='device to use for training / testing')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')

    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser.parse_args()

import os
import PIL
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def bulid_dataloader(is_train, args):
    dataset = build_dataset(is_train, args)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    return dataloader

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def build_cifar100_transform(is_train, input_size=224):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    if is_train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return transform

def build_cifar100_dataset_and_dataloader(is_train=True, batch_size=64, num_workers=2, input_size=224, args=None):
    transform = build_cifar100_transform(is_train, input_size)
    # transform = build_transform(is_train, args)
    dataset = datasets.CIFAR100(
        root='./data',
        train=is_train,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return dataloader



def imshow(img):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    img = img * np.array(std).reshape(3, 1, 1) + np.array(mean).reshape(3, 1, 1)  # 使用mean和std进行反标准化

    np_img = img.numpy()  # 转换为 numpy 数组
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  # 转置维度为 HWC 以适配 imshow (C, H, W) -> (H, W, C)
    plt.show()


if __name__ == '__main__':
    train_loader = build_cifar100_dataset_and_dataloader(is_train=True, batch_size=64, num_workers=4)
    val_loader = build_cifar100_dataset_and_dataloader(is_train=False, batch_size=64, num_workers=4)

    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(images)

        imshow(images[0])

        break



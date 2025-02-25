import torch
import torch.nn as nn
from config import get_args_parser
from utils.dataloader import build_cifar100_dataset_and_dataloader


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=1, out_features=1 + 12 + 12 + 12)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(in_features=1 + 12 + 12 + 12,
                                 out_features=1 + 12 + 12 + 12)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, latency):
        probs = self.sigmoid(self.linear2(self.act(self.linear1(latency))))
        # print(probs)
        sub_dim_probs = probs[0, 0]
        depth_list_probs = probs[0, 1:13]
        mlp_ratio_list_probs = probs[0, 13:25]
        mha_head_list_probs = probs[0, 25:37]

        sub_dim = self.get_sub_dim(sub_dim_probs)
        depth_list = self.get_depth_list(depth_list_probs)
        mlp_ratio_list = self.get_mlp_ratio_list(mlp_ratio_list_probs)
        mha_head_list = self.get_mha_head_list(mha_head_list_probs)

        return sub_dim, depth_list, mlp_ratio_list, mha_head_list, probs

    def get_sub_dim(self, probs):
        if probs<1/3:
            index = 0
        elif probs<2/3:
            index = 1
        else:
            index = 2
        dim_list = [256, 320, 384]
        return dim_list[index]

    def get_depth_list(self, probs):
        depth_list = []
        for i in range(12):
            if probs[i] < 1 / 5:
                index = 0
            else:
                index = 1
            if index == 1:
                depth_list.append(i)
        return depth_list

    def get_mlp_ratio_list(self, probs):
        ratio_list = [1, 2, 4]
        mlp_ratio_list = []
        for i in range(12):
            if probs[i] < 1 / 3:
                index = 0
            elif probs[i] < 2 / 3:
                index = 1
            else:
                index = 2

            mlp_ratio_list.append(ratio_list[index])
        return mlp_ratio_list

    def get_mha_head_list(self, probs):
        heads_list = [4, 5, 6]
        mha_head_list = []
        for i in range(12):
            if probs[i] < 1 / 3:
                index = 0
            elif probs[i] < 2 / 3:
                index = 1
            else:
                index = 2
            mha_head_list.append(heads_list[index])
        return mha_head_list


if __name__ == '__main__':
    args = get_args_parser()
    # valDataLoader = build_cifar100_dataset_and_dataloader(is_train=False, batch_size=args.batch_size, num_workers=args.num_workers, args=args)

    device = "cuda:5"
    router = Router()
    model = router.to(device)

    src = torch.rand((1, 1))
    src = src.to('cuda:5')
    sub_dim_probs, depth_list_probs, mlp_ratio_list_probs, mha_head_list_probs, probs = model(src)
    print(sub_dim_probs, depth_list_probs, mlp_ratio_list_probs, mha_head_list_probs)
    print(probs)


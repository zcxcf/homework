import torch
import torch.nn as nn
from config import get_args_parser
from utils.dataloader import build_cifar100_dataset_and_dataloader


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=1, out_features=1*3+12*2+12*3+12*3)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(in_features=1*3+12*2+12*3+12*3, out_features=1*3+12*2+12*3+12*3)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, latency):
        probs = self.sigmoid(self.linear2(self.act(self.linear1(latency))))
        # print(probs)
        sub_dim_probs = probs[:, 0:3]
        depth_list_probs = probs[:, 3:27]
        mlp_ratio_list_probs = probs[:, 27:63]
        mha_head_list_probs = probs[:, 63:99]

        sub_dim, dim_index = self.get_sub_dim(sub_dim_probs)
        depth_list, depth_index_list = self.get_depth_list(depth_list_probs)
        mlp_ratio_list, ratio_index_list = self.get_mlp_ratio_list(mlp_ratio_list_probs)
        mha_head_list, heads_index_list = self.get_mha_head_list(mha_head_list_probs)

        chosen_index = dim_index+depth_index_list+ratio_index_list+heads_index_list
        
        return sub_dim, depth_list, mlp_ratio_list, mha_head_list, probs[0][chosen_index], probs

    def get_sub_dim(self, probs):
        index_list = []
        index = torch.argmax(probs)
        index_list.append(index.item())
        dim_list = [256, 320, 384]
        return dim_list[index.item()], index_list

    def get_depth_list(self, probs):
        depth_list = []
        index_list = []
        for i in range(12):
            index = torch.argmax(probs[0][i*2:i*2+2])
            if index==0:
                depth_list.append(i)
            index_list.append(index.item()+i*2+3)
        return depth_list, index_list

    def get_mlp_ratio_list(self, probs):
        ratio_list = [1, 2, 4]
        mlp_ratio_list = []
        index_list = []
        for i in range(12):
            index = torch.argmax(probs[0][i*3:i*3+3])
            mlp_ratio_list.append(ratio_list[index])
            index_list.append(index.item()+i*3+27)
        return mlp_ratio_list, index_list

    def get_mha_head_list(self, probs):
        heads_list = [4, 5, 6]
        mha_head_list = []
        index_list = []
        for i in range(12):
            index = torch.argmax(probs[0][i*3:i*3+3])
            mha_head_list.append(heads_list[index])
            index_list.append(index.item()+i*3+63)
        return mha_head_list, index_list

if __name__ == '__main__':
    args = get_args_parser()
    # valDataLoader = build_cifar100_dataset_and_dataloader(is_train=False, batch_size=args.batch_size, num_workers=args.num_workers, args=args)

    device = "cuda:5"
    router = Router()
    model = router.to(device)

    src = torch.rand((1, 1))
    src = src.to('cuda:5')
    sub_dim_probs, depth_list_probs, mlp_ratio_list_probs, mha_head_list_probs, index_list = model(src)
    print(sub_dim_probs, depth_list_probs, mlp_ratio_list_probs, mha_head_list_probs)
    print(index_list)


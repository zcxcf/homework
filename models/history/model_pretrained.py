from functools import partial

import torch
import torch.nn as nn
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

import torch
import torch.nn as nn
from functools import partial
from timm.layers import Mlp as VitMlp
from timm.models.vision_transformer import Attention

from ptflops import get_model_complexity_info


model = timm.models.vision_transformer.vit_small_patch16_224(pretrained=True, num_classes=100)
model.to('cuda')
torch.save(model.state_dict(), '/home/ssd7T/zc_reuse/iccv/pretrained_para/vit_small.pth')

print(model)
src = torch.rand((1, 3, 224, 224))
src = src.to('cuda')
out = model(src)
print(out.shape)
print()


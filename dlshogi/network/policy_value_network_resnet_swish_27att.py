import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import *

class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(shape))

    def forward(self, input):
        return input + self.bias

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


fcl = 256  # fully connected layers

class PolicyValueNetwork(nn.Module):
    def __init__(self, resnet_blocks=10, k=192):
        super(PolicyValueNetwork, self).__init__()
        self.resnet_blocks = resnet_blocks
        self.k = k
        
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=k, kernel_size=1, bias=False)  # pieces_in_hand

        self.resnet_layers = nn.ModuleList()
        for _ in range(resnet_blocks):
            self.resnet_layers.append(nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False))
            self.resnet_layers.append(nn.BatchNorm2d(k))
            self.resnet_layers.append(nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False))
            self.resnet_layers.append(nn.BatchNorm2d(k))

        # policy network
        self.l22 = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.l22_2 = Bias(9*9*MAX_MOVE_LABEL_NUM)

        # value network
        self.l22_v = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.l23_v = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.l24_v = nn.Linear(fcl, 1)
        
        self.norm1 = nn.BatchNorm2d(k)
        self.norm22_v = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.swish = nn.SiLU()

        # policy attention branch
        self.att_l1_p = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=3, padding=1, bias=False)
        self.att_norm1_p = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.att_l2_p = nn.Conv2d(in_channels=MAX_MOVE_LABEL_NUM, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.att_norm2_p = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.att_l3_p = nn.Conv2d(in_channels=MAX_MOVE_LABEL_NUM, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.att_l3_p_2 = Bias(9*9*MAX_MOVE_LABEL_NUM)

        # value attention branch
        self.att_l1_v = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=3, padding=1, bias=False)
        self.att_norm1_v = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.att_l2_v = nn.Conv2d(in_channels=MAX_MOVE_LABEL_NUM, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.att_norm2_v = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.att_l3_v = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.att_l4_v = nn.Linear(fcl, 1)

    def forward(self, x1, x2):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u = self.swish(self.norm1(u1_1_1 + u1_1_2 + u1_2))

        for i in range(0, len(self.resnet_layers), 4):
            h1 = self.swish(self.resnet_layers[i+1](self.resnet_layers[i](u)))
            h2 = self.resnet_layers[i+3](self.resnet_layers[i+2](h1))
            u = self.swish(h2 + u)

        # policy attention branch
        att1_p = self.swish(self.att_norm1_p(self.att_l1_p(h2)))
        self.att_p = torch.sigmoid(self.att_norm2_p(self.att_l2_p(att1_p)))
        att3_p = self.att_l3_p_2(torch.flatten(self.att_l3_p(att1_p), 1))

        # policy network
        h22 = self.l22(u * self.att_p.mean(dim=1, keepdim=True) + u)
        h22_1 = self.l22_2(torch.flatten(h22, 1))

        # value attention branch
        att1_v = self.swish(self.att_norm1_v(self.att_l1_v(h2)))
        self.att_v = torch.sigmoid(self.att_norm2_v(self.att_l2_v(att1_v)))
        att3_v = self.att_l4_v(self.swish(self.att_l3_v(torch.flatten(att1_v, 1))))

        # value network
        h22_v = self.swish(self.norm22_v(self.l22_v(u * self.att_v.mean(dim=1, keepdim=True) + u)))
        h23_v = self.swish(self.l23_v(torch.flatten(h22_v, 1)))
        
        return h22_1, self.l24_v(h23_v), att3_p, att3_v

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self.swish = nn.SiLU() if memory_efficient else Swish()

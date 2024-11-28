import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import *

class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias=nn.Parameter(torch.Tensor(shape))

    def forward(self, input):
        return input + self.bias

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


k = 16
fcl = 256 # fully connected layers
class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=k, kernel_size=1, bias=False) # pieces_in_hand
        
        self.l1_3 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        
        self.l2 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l3 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        
        # self.l4 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l5 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l6 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l7 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l8 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l9 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l10 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l11 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l12 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l13 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l14 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l15 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l16 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l17 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l18 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l19 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l20 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # self.l21 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        
        # # policy network
        # self.l22 = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        # self.l22_2 = Bias(9*9*MAX_MOVE_LABEL_NUM)
        
        # value network
        self.l22_v = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.l23_v = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.l24_v = nn.Linear(fcl, 1)
        
        self.norm1 = nn.BatchNorm2d(k)
        self.norm2 = nn.BatchNorm2d(k)
        self.norm3 = nn.BatchNorm2d(k)
        
        # self.norm4 = nn.BatchNorm2d(k)
        # self.norm5 = nn.BatchNorm2d(k)
        # self.norm6 = nn.BatchNorm2d(k)
        # self.norm7 = nn.BatchNorm2d(k)
        # self.norm8 = nn.BatchNorm2d(k)
        # self.norm9 = nn.BatchNorm2d(k)
        # self.norm10 = nn.BatchNorm2d(k)
        # self.norm11 = nn.BatchNorm2d(k)
        # self.norm12 = nn.BatchNorm2d(k)
        # self.norm13 = nn.BatchNorm2d(k)
        # self.norm14 = nn.BatchNorm2d(k)
        # self.norm15 = nn.BatchNorm2d(k)
        # self.norm16 = nn.BatchNorm2d(k)
        # self.norm17 = nn.BatchNorm2d(k)
        # self.norm18 = nn.BatchNorm2d(k)
        # self.norm19 = nn.BatchNorm2d(k)
        # self.norm20 = nn.BatchNorm2d(k)
        # self.norm21 = nn.BatchNorm2d(k)
        
        self.norm22_v = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.swish = nn.SiLU()
        
        # input attention branch
        self.att_l0_1_p = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=3, padding=1, bias=False)
        self.att_norm0_1_p = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.att_l0_2_p = nn.Conv2d(in_channels=MAX_MOVE_LABEL_NUM, out_channels=1, kernel_size=1, bias=False)
        self.att_norm0_2_p = nn.BatchNorm2d(1)

        # # policy attention branch
        # self.att_l1_p = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=3, padding=1, bias=False)
        # self.att_norm1_p = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        # self.att_l2_p = nn.Conv2d(in_channels=MAX_MOVE_LABEL_NUM, out_channels=1, kernel_size=1, bias=False)
        # self.att_norm2_p = nn.BatchNorm2d(1)
        # self.att_l3_p = nn.Conv2d(in_channels=MAX_MOVE_LABEL_NUM, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        # self.att_l3_p_2 = Bias(9*9*MAX_MOVE_LABEL_NUM)
        
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
        u1 = self.swish(self.norm1(u1_1_1 + u1_1_2 + u1_2))
        
        # input attention branch
        att_0 = self.swish(self.att_norm0_1_p(self.att_l0_1_p(u1)))
        self.att_input = torch.sigmoid(self.att_norm0_2_p(self.att_l0_2_p(att_0)))
        u1_3 = self.l1_3(u1 * self.att_input + u1)
        
        # Residual block
        h2 = self.swish(self.norm2(self.l2(u1_3)))
        h3 = self.norm3(self.l3(h2))
        u3 = self.swish(h3 + u1)
        
        # # policy attention branch
        # att1_p = self.swish(self.att_norm1_p(self.att_l1_p(h3)))
        # self.att_p = torch.sigmoid(self.att_norm2_p(self.att_l2_p(att1_p)))
        # att3_p = self.att_l3_p_2(self.att_l3_p(att1_p).view(-1, 9*9*MAX_MOVE_LABEL_NUM))
        # # policy network
        # h22 = self.l22(u3 * self.att_p + u3)
        # h22_1 = self.l22_2(h22.view(-1, 9*9*MAX_MOVE_LABEL_NUM))
        
        # value attention branch
        att1_v = self.swish(self.att_norm1_v(self.att_l1_v(h3)))
        self.att_v = torch.sigmoid(self.att_norm2_v(self.att_l2_v(att1_v)))
        att3_v = self.att_l4_v(self.swish(self.att_l3_v(att1_v.view(-1, 9*9*MAX_MOVE_LABEL_NUM))))
        # value network
        h22_v = self.swish(self.norm22_v(self.l22_v(u3 * self.att_v.mean(dim=1, keepdim=True) + u3)))
        h23_v = self.swish(self.l23_v(h22_v.view(-1, 9*9*MAX_MOVE_LABEL_NUM)))
        return self.l24_v(h23_v), att3_v


    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self.swish = nn.SiLU() if memory_efficient else Swish()

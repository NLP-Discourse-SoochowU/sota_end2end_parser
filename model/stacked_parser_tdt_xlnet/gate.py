# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from config import SEED
import torch
import torch.nn as nn
from config import HIDDEN_SIZE, GateDrop, GATE_V, BOUND_INFO_SIZE
torch.manual_seed(SEED)


class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()
        self.wA = nn.Linear(HIDDEN_SIZE, 1)
        self.wB = nn.Linear(HIDDEN_SIZE, 1)
        # drop out
        self.gate_dropout = nn.Dropout(GateDrop)

    def forward(self, split_info, bu_info):
        """ Return 2 * HIDDEN_SIZE
        """
        if False:
            gated = torch.sigmoid((self.wA(self.gate_dropout(split_info)) + self.wB(self.gate_dropout(bu_info))))
            out_ = torch.cat(((1 - gated) * split_info, gated * bu_info), -1)
        else:
            # 2 个人的信息自我控制
            gate_a = torch.sigmoid(self.wA(self.gate_dropout(split_info)))
            gate_b = torch.sigmoid(self.wB(self.gate_dropout(bu_info)))
            out_ = torch.cat((gate_a * split_info, gate_b * bu_info), -1)
        return out_

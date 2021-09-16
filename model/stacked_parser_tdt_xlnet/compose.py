# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description:
"""
import torch
from config import *
import torch.nn as nn
torch.manual_seed(SEED)


class Compose(nn.Module):
    """ Desc: The composition function for reduce option.
    """
    def __init__(self):
        nn.Module.__init__(self)
        proj_in = HIDDEN_SIZE
        self.project = nn.Sequential(
            nn.Linear(proj_in, (HIDDEN_SIZE // 2) * 5),
            nn.Dropout(p=0.2)
        )
        self.left_meta = nn.Parameter(torch.empty(1, HIDDEN_SIZE, dtype=torch.float))
        self.right_meta = nn.Parameter(torch.empty(1, HIDDEN_SIZE, dtype=torch.float))
        nn.init.xavier_normal_(self.left_meta)
        nn.init.xavier_normal_(self.right_meta)

    def forward(self, left, right):
        h1, c1 = self.left_meta.squeeze(0).chunk(2) if left is None else left.chunk(2)
        h2, c2 = self.right_meta.squeeze(0).chunk(2) if right is None else right.chunk(2)
        hidden_states = torch.cat((h1, h2), -1)
        g, i, f1, f2, o = self.project(hidden_states).chunk(5)
        c = g.tanh() * i.sigmoid() + f1.sigmoid() * c1 + f2.sigmoid() * c2
        h = o.sigmoid() * c.tanh()
        return torch.cat([h, c])

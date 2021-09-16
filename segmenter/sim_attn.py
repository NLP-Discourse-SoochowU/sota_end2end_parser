# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from config_segment import *
import torch
import torch.nn as nn
torch.manual_seed(SEED)


class Sim_Attn(nn.Module):
    def __init__(self, encoder_size, decoder_size, num_labels=1, hidden_size=HIDDEN_SIZE):
        super(Sim_Attn, self).__init__()
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.e_mlp = nn.Sequential(
            nn.Linear(encoder_size, hidden_size, bias=False),
            nn.SELU(),
            nn.Linear(hidden_size, K, bias=False),
            nn.SELU()
        ) if MLP_Layer == 2 else nn.Sequential(
            nn.Linear(encoder_size, K, bias=False),
            nn.SELU()
        )
        self.d_mlp = nn.Sequential(
            nn.Linear(decoder_size, hidden_size, bias=False),
            nn.SELU(),
            nn.Linear(hidden_size, K, bias=False),
            nn.SELU()
        ) if MLP_Layer == 2 else nn.Sequential(
            nn.Linear(encoder_size, K, bias=False),
            nn.SELU()
        )
        self.u1 = nn.Parameter(torch.empty(K, K, dtype=torch.float))
        self.u2 = nn.Parameter(torch.empty(K, num_labels, dtype=torch.float))
        self.b = nn.Parameter(torch.empty(1, 1, num_labels, dtype=torch.float))
        nn.init.xavier_normal_(self.u1)
        nn.init.xavier_normal_(self.u2)
        nn.init.xavier_normal_(self.b)

    def forward(self, e_outputs, d_outputs):
        h_e = self.e_mlp(e_outputs)
        h_d = self.d_mlp(d_outputs)
        part1 = h_e.matmul(self.u1)
        part1 = part1.bmm(h_d.transpose(1, 2)).transpose(1, 2).unsqueeze(-1)
        part2 = h_e.matmul(self.u2).unsqueeze(1)
        s = part1 + part2 + self.b
        return s

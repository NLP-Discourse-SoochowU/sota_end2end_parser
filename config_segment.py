# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
VERSION, SET, USE_GPU, CUDA_ID = 10, 200, True, 3
HIDDEN_SIZE, Head_NUM, GCN_LAYER, K, RNN_LAYER, RNN_TYPE = 384, 2, 2, 64, 2, "GRU"
USE_ELMo, USE_POS = True, True
Partial_IN = False
TOKEN_SCORE = False
LR, LR_DECAY = 0.001, False
BATCH_SIZE, N_EPOCH, LOG_EVE, EVA_EVE = 1, 16, 20, 120
SAVE_MODEL = True
DROPOUT = 0.2
RESIDUAL_DROPOUT = 0.33
L2 = 1e-5
USE_ENC_DEC = True
BASELINE, Bias, SIM_BIAS, R = True, False, False, 32  # basic GCN
MLP_Layer = 1
SEED = 7
DEV_SIZE = 640
PAD = "<PAD>"
PAD_ID = 0
UNK = "<UNK>"
UNK_ID = 1
USE_ALL_SYN_INFO = False
tag2ids = {"O": 0, "B": 1, PAD: 2}
sync2ids = {"head": 0, "dep": 1, "self": 2}
TAG_LABELS = ["O", "B"]
SYN_SIZE = 81 if USE_ALL_SYN_INFO else 3
POS_TAG_NUM = 47
POS_TAG_SIZE = 30 if USE_POS else 0
WORDEMB_SIZE = 1024 if USE_ELMo else 300
ELMo_SIZE = 512
EMBED_LEARN = False
MAX_SEQ_LEN = 140
SMOO_VAL = -1e2
PRINT_EVE = 10000

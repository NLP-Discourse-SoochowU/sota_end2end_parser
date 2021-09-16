# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import random
import torch.nn.functional as func
import numpy as np
from config_segment import *
from path_config import IDS2VEC, DATA_SETS, DATA_SETS_SYN, LOG_ALL_SEG, MODEL_SAVE_SEG
import torch
from util.data_iterator import gen_batch_iter
from util.file_util import *
from segmenter.model import Segment_Model as Segment_Model_Stronger
import torch.optim as optim

random.seed(SEED)
torch.random.manual_seed(SEED)
np.random.seed(SEED)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def gen_loss(inputs, target, model):
    score_, attn = model(inputs)
    score_ = score_.squeeze(0).view(score_.size()[0] * score_.size()[1], -1)
    target = target.view(-1)
    loss_pred = func.nll_loss(score_, target)
    loss_difference = 0.
    for idx in range(Head_NUM):
        for c_idx in range(idx + 1, Head_NUM):
            tmp_loss = func.cosine_embedding_loss(attn[idx], attn[c_idx], target=torch.Tensor([-1]).cuda(CUDA_ID))
            loss_difference += tmp_loss
    loss_ = loss_pred + loss_difference
    return loss_


def main():
    log_file = os.path.join(LOG_ALL_SEG, "set_" + str(SET) + ".log")
    log_ite = 1
    word_emb = load_data(IDS2VEC)
    model = Segment_Model_Stronger(word_emb=word_emb)
    if USE_GPU:
        model.cuda(CUDA_ID)
    step, best_f1, max_b, macro_max_, loss_ = 0, 0, 0., 0., 0.
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=0.7, patience=17, min_lr=7e-5) \
        if LR_DECAY else None
    train_set, dev_set, test_set = load_data(DATA_SETS_SYN) if USE_ALL_SYN_INFO else load_data(DATA_SETS)
    for epoch in range(1, N_EPOCH):
        batch_iter = gen_batch_iter(train_set)
        for n_batch, (inputs, target) in enumerate(batch_iter, start=1):
            step += 1
            model.train()
            optimizer.zero_grad()
            tag_outputs, targets = target
            loss_ = loss_ + gen_loss(inputs, targets, model)
            if n_batch > 0 and n_batch % LOG_EVE == 0:
                loss_ = loss_ / float(LOG_EVE * BATCH_SIZE)
                loss_.backward()
                optimizer.step()
                print("VERSION: " + str(VERSION) + ", SET: " + str(SET) + " -- lr %f, epoch %d, batch %d, loss %.4f" %
                      (get_lr(optimizer), epoch, n_batch, loss_.item()))
                loss_ = 0.
                if n_batch % EVA_EVE == 0:
                    p_b, r_b, f_b = evaluate(dev_set, model)
                    if LR_DECAY:
                        scheduler.step(f_b)
                    if f_b > best_f1:
                        best_f1 = f_b
                        p_b_, r_b_, f_b_ = evaluate(test_set, model)
                        if f_b_ > macro_max_:
                            macro_max_ = f_b_
                            max_b = (p_b_, r_b_, f_b_)
                            check_str = check_max(max_b, n_epoch=epoch)
                            print_eve(ite=log_ite, str_=check_str, log_file=log_file)
                            log_ite += 1
                            if SAVE_MODEL:
                                torch.save(model, os.path.join(MODEL_SAVE_SEG, "EN_" + str(SET) + ".model"))


def check_max(max_b, n_epoch=0):
    (p_b, r_b, f_b) = max_b
    check_str = "---" + "VERSION: " + str(VERSION) + ", SET: " + str(SET) + ", EPOCH: " + str(n_epoch) + "---\n" + \
                "TEST (B): " + str(p_b) + "(P), " + str(r_b) + "(R), " + str(f_b) + "(F)\n"
    return check_str


def evaluate(dataset, model):
    model.eval()
    batch_iter = gen_batch_iter(dataset, batch_s=1)
    c_b, g_b, h_b = 0., 0., 0.
    for n_batch, (inputs, target) in enumerate(batch_iter, start=1):
        word_ids, word_elmo_embeddings, pos_ids, graph, _, _, masks = inputs
        targets = target[1].squeeze(0)
        pred = model.predict_(word_ids, word_elmo_embeddings, pos_ids, graph, masks)
        predict = pred.data.cpu().numpy()
        targets = targets.data.cpu().numpy()
        c_b += len(np.intersect1d(predict, targets))
        g_b += len(targets)
        h_b += len(predict)
        c_b -= 1
        g_b -= 1
        h_b -= 1
    p_b = 0. if h_b == 0 else c_b / h_b
    r_b = 0. if g_b == 0 else c_b / g_b
    f_b = 0. if (g_b + h_b) == 0 else (2 * c_b) / (g_b + h_b)
    result = (p_b, r_b, f_b)
    return result

# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import torch
import random
import numpy as np
from path_config import *
from config_segment import *
import progressbar
from util.file_util import *
from segmenter.sentence import Sentence
p = progressbar.ProgressBar()
p_1 = progressbar.ProgressBar()
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if USE_ELMo:
    from allennlp.modules.elmo import batch_to_ids
    from allennlp.modules.elmo import Elmo
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/" \
                   "elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo" \
                  "_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, 2, dropout=0)
ELMO_ROOT_PAD = torch.zeros(1, 1024)


def build_data_sets():
    sents_training = build_specific_data_set(RAW_TRAIN_FILES)
    sents_test = build_specific_data_set(RAW_TEST_FILES)
    data_set_raw = (sents_training, sents_test)
    save_data(data_set_raw, DATA_SETS_RAW)


def build_specific_data_set(file_path):
    data_set = list()
    p.start(385)
    p_idx = 1
    for file_ in os.listdir(file_path):
        if file_.endswith(".out"):
            p.update(p_idx)
            p_idx += 1
            sent_file_ = os.path.join(file_path, file_)
            edu_file_ = sent_file_ + ".edus"
            sent_txt_list, edu_txt_list = [], []
            with open(sent_file_, "r") as f:
                for line in f:
                    line = line.strip().lower()
                    if len(line) == 0:
                        continue
                    sent_txt_list.append(line)
            with open(edu_file_, "r") as f:
                for line in f:
                    line = line.strip().lower()
                    if len(line) == 0:
                        continue
                    edu_txt_list.append(line)
            # sentence 2 edus
            p_1.start(len(sent_txt_list))
            p1_idx = 1
            while len(sent_txt_list) > 0:
                p_1.update(p1_idx)
                p1_idx += 1
                sent = sent_txt_list.pop(0)
                edus_list = []
                tmp_edu = ""
                tmp_sent = "".join(sent.split())
                while len(tmp_edu) < len(tmp_sent):
                    tmp_edu_txt = edu_txt_list.pop(0)
                    edus_list.append(tmp_edu_txt)
                    tmp_edu += "".join(tmp_edu_txt.split())
                    while len(tmp_edu) > len(tmp_sent):
                        p_1.update(p1_idx)
                        p1_idx += 1
                        sent = sent + " " + sent_txt_list.pop(0)
                        tmp_sent = "".join(sent.split())
                data_set.append(Sentence(sent, edus_list))
    p.finish()
    return data_set


def build_voc(data_set):
    words_set = set()
    with open(GLOVE200, "r") as f:
        for line in f:
            tokens = line.split()
            words_set.add(tokens[0])
    # build word2ids
    word2ids, pos2ids, syn2ids = dict(), dict(), dict()
    word2freq = dict()
    word2ids[PAD], word2ids[UNK] = 0, 1
    pos2ids[PAD], pos2ids[UNK] = 0, 1
    idx_1, idx_2, idx_3 = 2, 2, 0
    sents_training, sents_test = data_set
    total_sents = sents_training + sents_test
    for sent in total_sents:
        for edu in sent.edus:
            for word, pos_tag in zip(edu.words, edu.pos_tags):
                if word not in word2freq.keys():
                    word2freq[word] = 1
                elif word not in word2ids.keys() and word in words_set:
                    word2freq[word] += 1
                    word2ids[word] = idx_1
                    idx_1 += 1
                else:
                    word2freq[word] += 1
                if pos_tag not in pos2ids.keys():
                    pos2ids[pos_tag] = idx_2
                    idx_2 += 1
        for dep_pair in sent.dependency:
            head_rel = dep_pair[0] + "-head"
            if head_rel not in syn2ids.keys():
                syn2ids[head_rel] = idx_3
                idx_3 += 1
            dep_rel = dep_pair[0] + "-dep"
            if dep_rel not in syn2ids.keys():
                syn2ids[dep_rel] = idx_3
                idx_3 += 1
    syn2ids["self"] = idx_3
    for word in word2freq.keys():
        if word not in word2ids.keys():
            word2ids[word] = word2ids[UNK]
    save_data(word2ids, WORD2IDS)
    save_data(pos2ids, POS2IDS)
    save_data(syn2ids, SYN2IDS)
    build_ids2_vec()


def build_ids2_vec():
    word2ids = load_data(WORD2IDS)
    ids2vec = dict()
    with open(GLOVE200, "r") as f:
        for line in f:
            tokens = line.split()
            word = tokens[0]
            vec = np.array([[float(token) for token in tokens[1:]]])
            if tokens[0] in word2ids.keys() and word2ids[tokens[0]] != UNK_ID:
                ids2vec[word2ids[word]] = vec
    # transform into numpy array
    # PAD and UNK
    embed = [np.zeros(shape=(WORDEMB_SIZE,), dtype=np.float32)]
    embed = np.append(embed, [np.random.uniform(-0.25, 0.25, WORDEMB_SIZE)], axis=0)
    # others
    idx_valid = list(ids2vec.keys())
    idx_valid.sort()
    for idx in idx_valid:
        embed = np.append(embed, ids2vec[idx], axis=0)
    save_data(embed, IDS2VEC)


def build_data_ids():
    data_set_raw = load_data(DATA_SETS_RAW)
    data_set = gen_instances(data_set_raw)
    if USE_ALL_SYN_INFO:
        save_data(data_set, DATA_SETS_SYN)
    else:
        save_data(data_set, DATA_SETS)


def gen_instances(data_sets_raw):
    """ Load sentences and EDUs from source files and generate the data list with ids.
        (word_ids, pos_ids, syn_ids, tag_ids)
    """
    word2ids, pos2ids, syn2ids = load_data(WORD2IDS), load_data(POS2IDS), load_data(SYN2IDS)
    # train dev and test sets
    train_set, test_set = data_sets_raw[0], data_sets_raw[1]
    # randomly sample
    dev_set = list()
    random.shuffle(train_set)
    for idx in range(DEV_SIZE):
        dev_set.append(train_set.pop(0))
    train_list = gen_specific_instances(train_set, word2ids, pos2ids, syn2ids)
    test_list = gen_specific_instances(test_set, word2ids, pos2ids, syn2ids)
    dev_list = gen_specific_instances(dev_set, word2ids, pos2ids, syn2ids)
    return train_list, dev_list, test_list


def gen_specific_instances(data_set, word2ids, pos2ids, syn2ids):
    """ Transform all data into ids.
        We take root node into consideration.
    """
    print("check the server!")
    p_2 = progressbar.ProgressBar()
    p_2.start(len(data_set))
    p2_idx = 1
    data_set_ = []
    for sentence in data_set:
        p_2.update(p2_idx)
        p2_idx += 1
        edus = sentence.edus
        sent_words_ids, sent_poses_ids, sent_tags_ids, graph_ids = [], [], [], []
        sent_token_list = []
        for i, edu in enumerate(edus):
            words = [word2ids[word] for word in edu.words]
            poses = [pos2ids[pos] for pos in edu.pos_tags]
            tags = ['O'] * (len(words) - 1)
            tags += ['B'] if i < len(edus) - 1 else ['O']
            tags = [tag2ids[tag] for tag in tags]
            sent_words_ids.extend(words)
            sent_poses_ids.extend(poses)
            sent_tags_ids.extend(tags)
            sent_token_list += edu.words
        # root PAD
        sent_words_ids.insert(0, PAD_ID)
        sent_poses_ids.insert(0, PAD_ID)
        sent_tags_ids.insert(0, 0)

        # token2elmo
        if USE_ELMo:
            sents_tokens_ids = batch_to_ids([sent_token_list])  # (1, sent_len)
            tmp_sent_tokens_emb = elmo(sents_tokens_ids)["elmo_representations"][0][0]
            tmp_sent_tokens_emb = torch.cat((ELMO_ROOT_PAD, tmp_sent_tokens_emb), 0)
        else:
            tmp_sent_tokens_emb = None

        # (type, "head", "dep")
        for i, dep_pair in enumerate(sentence.dependency):
            if USE_ALL_SYN_INFO:
                graph_ids.append((i, i, syn2ids["self"]))
                graph_ids.append((dep_pair[1], dep_pair[2], syn2ids[dep_pair[0] + "-head"]))
                graph_ids.append((dep_pair[2], dep_pair[1], syn2ids[dep_pair[0] + "-dep"]))
            else:
                graph_ids.append((i, i, sync2ids["self"]))
                graph_ids.append((dep_pair[1], dep_pair[2], sync2ids["head"]))
                graph_ids.append((dep_pair[2], dep_pair[1], sync2ids["dep"]))
        data_set_.append((sent_words_ids, sent_poses_ids, graph_ids, sent_tags_ids, tmp_sent_tokens_emb))
    p_2.finish()
    return data_set_


if __name__ == "__main__":
    # build_data_sets()
    build_data_ids()

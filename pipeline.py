# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description: Implementation of ELMo and bert in RST-style Segmentation.
"""
from util.file_util import *
from config_segment import *
from config import UNK_ids
from path_config import *
from stanfordcorenlp import StanfordCoreNLP
from allennlp.modules.elmo import batch_to_ids
from allennlp.modules.elmo import Elmo
import numpy as np
import torch
from transformers import *
import progressbar
from structure.rst_tree import rst_tree
from config import ids2nr, XLNET_TYPE, USE_CUDA, CUDA_ID
import gc

p = progressbar.ProgressBar()

tokenizer_xl = XLNetTokenizer.from_pretrained(XLNET_TYPE)
model_xl = torch.load("data/models_saved/v9_set149/test_f_max_xl_model.pth")
model_xl.eval()
if USE_CUDA:
    model_xl.cuda(CUDA_ID)

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/" \
               "elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo" \
              "_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo = Elmo(options_file, weight_file, 2, dropout=0)
path_to_jar = 'stanford-corenlp-full-2018-02-27'
nlp = StanfordCoreNLP(path_to_jar)
word2ids, pos2ids, syn2ids = load_data(WORD2IDS), load_data(POS2IDS), load_data(SYN2IDS)
ELMO_ROOT_PAD = torch.zeros(1, 1024)
p = progressbar.ProgressBar()


class PartitionPtrParser:
    def __init__(self):
        self.ids2nr = ids2nr

    def parse(self, instances, model):
        if len(instances) == 1:
            tree_parsed = rst_tree(temp_edu=instances[0])
        else:
            session = model.init_session(instances, model_xl, tokenizer_xl)
            d_masks, splits = None, []
            while not session.terminate():
                split_score, nr_score, state, d_mask = model.parse_predict(session)
                d_masks = d_mask if d_masks is None else torch.cat((d_masks, d_mask), 1)
                split = split_score.argmax()
                nr = self.ids2nr[nr_score[split].argmax()]
                nuclear, relation = nr.split("-")[0], "-".join(nr.split("-")[1:])
                session = session.forward(split_score, state, split, nuclear, relation)
            # build tree by splits (left, split, right)
            tree_parsed = self.build_rst_tree(instances, session.splits[:], session.nuclear[:], session.relations[:])
            # self.traverse_tree(tree_parsed)
        return tree_parsed

    def build_rst_tree(self, edus, splits, nuclear, relations, type_="Root", rel_=None):
        left, split, right = splits.pop(0)
        nucl = nuclear.pop(0)
        rel = relations.pop(0)
        left_n, right_n = nucl[0], nucl[1]
        left_rel = rel if left_n == "N" else "span"
        right_rel = rel if right_n == "N" else "span"

        if right - split == 0:
            # leaf node
            right_node = rst_tree(temp_edu=edus[split + 1][0], type_=right_n, rel=right_rel)
        else:
            # non leaf
            right_node = self.build_rst_tree(edus, splits, nuclear, relations, type_=right_n, rel_=right_rel)

        if split - left == 0:
            # leaf node
            left_node = rst_tree(temp_edu=edus[split][0], type_=left_n, rel=left_rel)
        else:
            # none leaf
            left_node = self.build_rst_tree(edus, splits, nuclear, relations, type_=left_n, rel_=left_rel)

        root = rst_tree(l_ch=left_node, r_ch=right_node, ch_ns_rel=nucl, child_rel=rel, type_=type_, rel=rel_)
        return root

    def traverse_tree(self, root):
        if root.left_child is not None:
            self.traverse_tree(root.left_child)
            self.traverse_tree(root.right_child)
            print("Inner: ", root.type, root.rel, root.temp_edu, root.child_rel, root.child_NS_rel)
        else:
            print("Leaf: ", root.type, root.rel, root.temp_edu)

    def draw_scores_matrix(self, model):
        scores = model.scores
        self.draw_decision_hot_map(scores)

    @staticmethod
    def draw_decision_hot_map(scores):
        import matplotlib
        import matplotlib.pyplot as plt
        text_colors = ["black", "white"]
        c_map = "YlGn"
        y_label = "split score"
        col_labels = ["split %d" % i for i in range(0, scores.shape[1])]
        row_labels = ["step %d" % i for i in range(1, scores.shape[0] + 1)]
        fig, ax = plt.subplots()
        im = ax.imshow(scores, cmap=c_map)
        c_bar = ax.figure.colorbar(im, ax=ax)
        c_bar.ax.set_ylabel(y_label, rotation=-90, va="bottom")
        ax.set_xticks(np.arange(scores.shape[1]))
        ax.set_yticks(np.arange(scores.shape[0]))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_xticks(np.arange(scores.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(scores.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        threshold = im.norm(scores.max()) / 2.
        val_fmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")
        texts = []
        kw = dict(horizontalalignment="center", verticalalignment="center")
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                kw.update(color=text_colors[im.norm(scores[i, j]) > threshold])
                text = im.axes.text(j, i, val_fmt(scores[i, j], None), **kw)
                texts.append(text)
        fig.tight_layout()
        plt.show()


def prep_seg(dt_path=None):
    with open(dt_path, "r") as f:
        sentences = f.readlines()
    sents_dt = []
    for idx, sent in enumerate(sentences):
        sent = sent.strip()
        if len(sent) == 0:
            continue
        tok_pairs = nlp.pos_tag(sent.strip())
        words = [pair[0] for pair in tok_pairs]
        tags = [pair[1] for pair in tok_pairs]
        word_ids = []
        for word in words:
            if word.lower() in word2ids.keys():
                word_ids.append(word2ids[word.lower()])
            else:
                word_ids.append(UNK_ids)
        pos_ids = [pos2ids[tag] for tag in tags]

        word_ids.insert(0, PAD_ID)
        pos_ids.insert(0, PAD_ID)

        graph_ids = []
        dependency = nlp.dependency_parse(sent)
        # (type, "head", "dep")
        for i, dep_pair in enumerate(dependency):
            graph_ids.append((i, i, sync2ids["self"]))
            graph_ids.append((dep_pair[1], dep_pair[2], sync2ids["head"]))
            graph_ids.append((dep_pair[2], dep_pair[1], sync2ids["dep"]))
        elmo_ids = batch_to_ids([words])
        tmp_sent_tokens_emb = elmo(elmo_ids)["elmo_representations"][0][0]
        tmp_sent_tokens_emb = torch.cat((ELMO_ROOT_PAD, tmp_sent_tokens_emb), 0)
        sents_dt.append((words, word_ids, pos_ids, graph_ids, None, tmp_sent_tokens_emb))
    return sents_dt


def do_seg(sents_dt_, rt_path=None):
    result_dt = [sents_dt_]
    # segment
    segmenter = torch.load(os.path.join(MODEL_SAVE_SEG, "EN_200.model"))
    segmenter.eval()
    segmenter.cuda(CUDA_ID)
    edus_all = []
    for doc_dt in result_dt:
        batch_iter = gen_batch_iter(doc_dt, batch_s=1)
        for n_batch, inputs in enumerate(batch_iter, start=1):
            words_all, word_ids, word_elmo_embeddings, pos_ids, graph, masks = inputs
            pred = segmenter.predict_(word_ids, word_elmo_embeddings, pos_ids, graph, masks)
            predict = pred.data.cpu().numpy()
            # transform to EDUs
            words_all = words_all[0]
            edus_all += fetch_edus(words_all, predict)
        edus_all.append("")
    # write to file
    write_iterate(edus_all, rt_path, append_=True)
    return edus_all


def fetch_edus(words_all, predict):
    edus_all = []
    tmp_edu = ""
    pred_idx = 0
    tmp_pre = predict[pred_idx]
    for idx, word in enumerate(words_all):
        if idx == tmp_pre:
            tmp_edu = tmp_edu.strip()
            edus_all.append(tmp_edu)
            tmp_edu = ""
            pred_idx += 1
            if pred_idx < predict.shape[0]:
                tmp_pre = predict[pred_idx]
        tmp_edu += (word + " ")
    tmp_edu = tmp_edu.strip()
    edus_all.append(tmp_edu)
    return edus_all


def gen_batch_iter(random_instances, batch_s=BATCH_SIZE):
    """ a batch 2 numpy data.
    """
    num_instances = len(random_instances)
    offset = 0
    while offset < num_instances:
        batch = random_instances[offset: min(num_instances, offset + batch_s)]
        num_batch = len(batch)
        lengths = np.zeros(num_batch, dtype=np.int)
        for i, (_, word_ids, _, _, _, _) in enumerate(batch):
            lengths[i] = len(word_ids)
        max_seq_len = lengths.max()
        # if max_seq_len >= MAX_SEQ_LEN:
        #     offset = offset + batch_s
        #     continue
        words_all, word_inputs, word_elmo_embeds, pos_inputs, graph_inputs, masks \
            = data_ids_prep(num_batch, max_seq_len, batch)
        offset = offset + batch_s
        # numpy2torch
        word_inputs = torch.from_numpy(word_inputs).long()
        word_elmo_embeds = torch.from_numpy(word_elmo_embeds).float()
        pos_inputs = torch.from_numpy(pos_inputs).long()
        graph_inputs = torch.from_numpy(graph_inputs).byte()
        masks = torch.from_numpy(masks).byte()
        if USE_GPU:
            word_inputs = word_inputs.cuda(CUDA_ID)
            word_elmo_embeds = word_elmo_embeds.cuda(CUDA_ID)
            pos_inputs = pos_inputs.cuda(CUDA_ID)
            graph_inputs = graph_inputs.cuda(CUDA_ID)
            masks = masks.cuda(CUDA_ID)
        yield words_all, word_inputs, word_elmo_embeds, pos_inputs, graph_inputs, masks


def data_ids_prep(num_batch, max_seq_len, batch):
    """ Transform all the data into the form of ids.
    """
    words_all = []
    word_inputs = np.zeros([num_batch, max_seq_len], dtype=np.long)
    word_elmo_embeddings = np.zeros([num_batch, max_seq_len, 1024], dtype=np.float)
    pos_inputs = np.zeros([num_batch, max_seq_len], dtype=np.long)
    graph_inputs = np.zeros([num_batch, max_seq_len, max_seq_len, SYN_SIZE], np.uint8)
    masks = np.zeros([num_batch, max_seq_len], dtype=np.uint8)
    for i, (words, word_ids, pos_ids, graph_ids, _, lm_embeds) in enumerate(batch):
        # word_ids, pos_ids, graph_ids, None, tmp_sent_tokens_emb
        words_all.append(words)
        seq_len = len(word_ids)
        word_inputs[i][:seq_len] = word_ids[:]
        word_elmo_embeddings[i][:seq_len][:] = lm_embeds.detach().numpy()
        pos_inputs[i][:seq_len] = pos_ids[:]
        for x, y, z in graph_ids:
            # Use one-hot vector to represent the connection between nodes, 0 denotes no, 1 refers to yes.
            graph_inputs[i, x, y, z] = 1
        masks[i][:seq_len] = 1
    return words_all, word_inputs, word_elmo_embeddings, pos_inputs, graph_inputs, masks


def prepare_dt(seg_edus):
    lines = seg_edus
    trees = []
    tmp_tree = []
    for line in lines:
        if len(line.strip()) == 0 and len(tmp_tree) > 0:
            trees.append(tmp_tree)
            tmp_tree = []
        else:
            tmp_tree.append(line.strip())
    if len(tmp_tree) > 0:
        trees.append(tmp_tree)

    instances = []
    for tree in trees:
        edus = tree
        encoder_inputs = []
        for edu in edus:
            edu_ = edu
            edu_word_ids = None
            edu_pos_ids = None
            edu_elmo_embeddings = None
            # boundary
            tmp_line = edu.strip()
            if tmp_line.endswith(".") or tmp_line.endswith("?") or tmp_line.endswith("!"):
                bound_info = 1
            else:
                bound_info = 0
            encoder_inputs.append((edu_, edu_word_ids, edu_elmo_embeddings, edu_pos_ids, bound_info))
        instances.append(encoder_inputs)
    return instances


def do_parse(seg_edus):
    edus = prepare_dt(seg_edus)
    model = torch.load("data/models_saved/v9_set149/test_f_max_model.pth").cuda(CUDA_ID)
    model.eval()
    parser = PartitionPtrParser()
    trees = []
    p.start(len(edus))
    p_idx = 1
    save_idx = 1
    for idx, doc_instances in enumerate(edus):
        p.update(p_idx)
        p_idx += 1
        tree = parser.parse(doc_instances, model)
        trees.append(tree)
        if idx > 0 and idx % 3000 == 0:
            save_data(trees, NMT_Trees_p + str(save_idx) + ".pkl")
            save_idx += 1
            del trees
            gc.collect()
            trees = []
    p.finish()
    return trees


if __name__ == "__main__":
    # segmenting
    sents_dt = prep_seg(raw_dt)
    seg_edus = do_seg(sents_dt, edu_dt)
    
    # parsing
    trees_ = do_parse(seg_edus)
    save_data(trees_, "data/e2e/trees.pkl")

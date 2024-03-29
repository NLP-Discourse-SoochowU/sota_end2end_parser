# coding: UTF-8
import numpy as np
import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as func
from config import *
from threading import Thread, current_thread
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.stacked_parser_tdt_xlnet.gate import Gate
from model.stacked_parser_tdt_xlnet.compose import Compose
import os

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class PartitionPtr(nn.Module):
    def __init__(self, word2ids, pos2ids, nuc2ids, rel2ids, pretrained):
        super(PartitionPtr, self).__init__()
        self.word2ids, self.pos2ids, self.nuc2ids, self.rel2ids = word2ids, pos2ids, nuc2ids, rel2ids
        if not (USE_ELMo or TRAIN_XLNET):
            self.word_emb = nn.Embedding(pretrained.shape[0], EMBED_SIZE)
            # nre_pre = np.array([arr[0:3] for arr in pretrained])
            self.word_emb.weight.data.copy_(torch.from_numpy(pretrained))
            self.word_emb.weight.requires_grad = True if EMBED_LEARN else False
        # self.pos_emb = nn.Embedding(len(pos2ids.keys()), POS_SIZE)
        self.bound_emb = nn.Embedding(3, BOUND_INFO_SIZE)
        # component
        self.edu_encoder = nn.GRU(1024, HIDDEN_SIZE, batch_first=True, bidirectional=True)
        # self.edu_encoder = BiGRUEDUEncoder(EMBED_SIZE + POS_SIZE) if USE_POSE else BiGRUEDUEncoder(EMBED_SIZE)
        self.context_encoder = Context_Encoder(HIDDEN_SIZE + BOUND_INFO_SIZE) if USE_BOUND else Context_Encoder(HIDDEN_SIZE)
        # self.context_dense = nn.Linear(self.context_encoder.output_size, HIDDEN_SIZE)
        self.decoder = Decoder(self.context_encoder.output_size * 2)
        self.split_attention = SplitAttention(self.context_encoder.output_size, self.decoder.output_size)
        # self.nr_scorer = nn.Linear(HIDDEN_SIZE, NR_LABEL_NUM)
        self.nr_classifier = BiAffineAttention(self.context_encoder.output_size, self.decoder.output_size,
                                               NR_LABEL_NUM, NR_MLP_SIZE)
        self.t_outputs = dict()
        self.comp = Compose()
        self.gate = Gate()
        self.gate_dense = nn.Linear(2 * HIDDEN_SIZE, HIDDEN_SIZE)
        self.residual_drop = nn.Dropout(Re_DROP)
        self.nr_linear = nn.Sequential(nn.Linear(NR_LABEL_NUM, 1),
                                       nn.Sigmoid())
        # Up—sampling
        self.down = nn.Sequential(nn.Conv2d(in_channel_G, out_channel_G, (ker_h_G, ker_w_G), strip_G),
                                  nn.ReLU())
        self.max_p = nn.MaxPool2d(kernel_size=(p_w_G, p_h_G), stride=p_w_G)
        dense_in = HIDDEN_SIZE * 2 if EDU_ENCODE_VERSION == 1 else XLNET_SIZE
        self.input_dense = nn.Linear(dense_in, HIDDEN_SIZE)

    def cnn_feat_ext(self, img):
        out = self.down(img)
        if MAX_POOLING:
            out = self.max_p(out)
        return out

    class Session:
        def __init__(self, memory, state, edus_h, splits_h, lengths, e_out_masks, bound_h):
            """ :param memory: 即为context_encoder编码结果（多输出）
                :param state: 是上下文信息降维结果
            """
            self.n = memory.size(1)  # (batch, cut_points_number, hidden)，n个切割点
            self.step = 0  # 当前步骤为第0步，还没有执行切割
            self.memory, self.state, self.edus_h, self.splits_h = memory, state, edus_h, splits_h
            self.lengths = lengths
            self.e_out_masks = e_out_masks
            # 初始化栈，记录当前切割区域，也是decoder的输入信息存储的地方
            self.stack = [(0, self.n - 1)]
            # score_table: (n个步骤, n+2个切割点) 为作图做准备
            self.scores = np.zeros((self.n, self.n), dtype=np.float)
            # 解码结果存储
            self.splits, self.nuclear, self.relations = [], [], []
            self.span_boundaries = None
            self.bound_h = bound_h

        def terminate(self):
            return self.step >= self.n

        def __repr__(self):
            return "[step %d]memory size: %s, state size: %s\n stack:\n%s\n, scores:\n %s" % \
                   (self.step, str(self.memory.size()), str(self.state.size()),
                    "\n".join(map(str, self.stack)) or "[]",
                    str(self.scores))

        def __str__(self):
            return repr(self)

        def forward(self, split_score, state, split, nuclear=None, relation=None):
            """ 实施预测结果，形成新的状态并记录
                累加需要分析的状态，注意解码是先右边，后左边，所以这里对状态信息存储采取先左，后右边。出栈从顶端开始
            """
            left, right = self.stack.pop()
            boundary_one = torch.LongTensor([left, right]).unsqueeze(0)
            self.span_boundaries = boundary_one if self.span_boundaries is None \
                else torch.cat((self.span_boundaries, boundary_one), 0)
            if split - left > 0:
                self.stack.append((left, split - 1))
            if right - split > 0:
                self.stack.append((split + 1, right))
            self.splits.append((left, split, right))
            self.nuclear.append(nuclear)
            self.relations.append(relation)
            self.state = state
            self.scores[self.step] = split_score
            self.step += 1
            return self

    def init_session(self, instances, model_xl=None, tokenizer_xl=None):
        """ 初始化状态信息，包括解码器中的 stack 信息
        """
        self.decoder.init_stack(left=0, right=len(instances) - 1)
        edu_num = len(instances)
        e_txt, e_txt_ = [], []
        e_in_words = e_in_word_embeds = e_in_poses = None
        e_bound_info = np.zeros([1, edu_num - 1], dtype=np.long)

        max_word_len = max(len(instance[0].strip().split()) for instance in instances)
        e_in_masks = np.zeros((1, edu_num, max_word_len), dtype=np.long)

        for i, edu in enumerate(instances):
            edu_, _, _, _, bound_info = edu
            e_txt_.append(edu_)
            if i < edu_num - 1:
                e_bound_info[0, i] = bound_info
            e_in_masks[0, i, :len(edu_.strip().split())] = 1
        e_in_masks = torch.from_numpy(e_in_masks).byte()
        if USE_CUDA:
            e_in_masks = e_in_masks.cuda(CUDA_ID)
        e_txt.append(e_txt_)
        # numpy2torch
        e_bound_info = torch.from_numpy(e_bound_info).long()
        if USE_CUDA:
            e_bound_info = e_bound_info.cuda(CUDA_ID)
        # edu 编码
        e_in = (e_txt, e_in_words, e_in_word_embeds, e_in_poses, e_in_masks)
        e_h, e_masks = self.encode_edus(e_in, model_xl, tokenizer_xl)
        if SPLIT_V == 0:
            splits_h, e_out_masks, lengths = self.context_encoder.get_splits(e_h, e_masks)
        elif SPLIT_V == 1:
            splits_h, e_out_masks, lengths = self.context_encoder.sim_split1(e_h, e_masks)
        else:
            splits_h, e_out_masks, lengths = self.context_encoder.sim_split2(e_h, e_masks)
        if USE_BOUND:
            bound_h = self.bound_emb(e_bound_info)
            splits_bounded = torch.cat((splits_h, bound_h), -1)
            memory, context = self.context_encoder(splits_bounded, lengths, e_out_masks)
        else:
            memory, context = self.context_encoder(splits_h, lengths, e_out_masks)
            bound_h = None
        return PartitionPtr.Session(memory, context, e_h, splits_h, lengths, e_out_masks, bound_h)

    def encode_edus(self, e_inputs, model_xl, tokenizer_xl):
        if EDU_ENCODE_VERSION == 0:
            return self.encode_edus1(e_inputs, model_xl, tokenizer_xl)
        elif EDU_ENCODE_VERSION == 1:
            return ...
        else:
            return self.encode_edus3(e_inputs, model_xl, tokenizer_xl)

    def encode_edus1(self, e_inputs, model_xl, tokenizer_xl):
        """ 方案 1：EDU XLNET 编码直接取平均作为 EDU 表征。
            其中，XLNET 的用法，1 Long 向量； 4 Float 向量.
        """
        e_txt, e_input_words, e_in_word_embeds, e_input_poses, e_masks = e_inputs
        batch_size, max_edu_len, max_word_len = e_input_words.size()
        e_masks = (e_masks.sum(-1) > 0).int()  # EDU mask, padded with the largest number of EDUs of different documents
        e_x = torch.zeros([1, XLNET_SIZE]).cuda(CUDA_ID)  # pad 向量
        edu_encoded_x = None
        for e_in_edus in e_txt:
            edu_x_ = None
            for idx in range(max_edu_len):
                if idx < len(e_in_edus):
                    edu__ = e_in_edus[idx]
                    input_ids = torch.tensor([tokenizer_xl.encode(edu__, add_special_tokens=True)]).cuda(CUDA_ID)  # (1, 6)
                    outputs = model_xl(input_ids)[0]
                    # tmp_edu = torch.mean(outputs[0], -2).cuda(CUDA_ID)  # (1, 1, 1024)
                    tmp_edu = outputs.squeeze()[-1].unsqueeze(0)
                    input_ids = input_ids.cpu()
                else:
                    tmp_edu = e_x
                edu_x_ = tmp_edu if edu_x_ is None else torch.cat((edu_x_, tmp_edu), 0)
            edu_x_ = edu_x_.unsqueeze(0)  # (1, edu_num, edu_embedding)
            edu_encoded_x = edu_x_ if edu_encoded_x is None else torch.cat((edu_encoded_x, edu_x_), 0)
        edu_encoded_x = self.input_dense(edu_encoded_x)
        return edu_encoded_x, e_masks

    # def encode_edus2(self, e_inputs, model_xl, tokenizer_xl):
    #     """ RNN based
    #     """
    #     e_txt, e_input_words, e_in_word_embeds, e_input_poses, e_masks = e_inputs
    #     batch_size, max_edu_len, max_word_len = e_input_words.size()
    #     e_masks = (e_masks.sum(-1) > 0).int()
    #     e_x = torch.zeros([1, 1024]).float().cuda(CUDA_ID)  # pad 向量
    #     edu_encoded_x = None
    #     for e_in_edus in e_txt:
    #         edu_x_ = None
    #         for idx in range(max_edu_len):
    #             if idx < len(e_in_edus):
    #                 edu__ = e_in_edus[idx]
    #                 encoded_ids = tokenizer_xl.encode(edu__, add_special_tokens=True)
    #                 input_ids = torch.tensor([encoded_ids]).cuda(CUDA_ID)  # (1, 6)
    #
    #                 outputs = model_xl(input_ids)[0].cuda(CUDA_ID)
    #                 tmp_edu = self.edu_encoder(outputs)[0]
    #                 tmp_edu = torch.mean(tmp_edu, -2)  # .float()  # (1, 1, 1024)
    #             else:
    #                 tmp_edu = e_x
    #             edu_x_ = tmp_edu if edu_x_ is None else torch.cat((edu_x_, tmp_edu), 0)
    #         edu_x_ = edu_x_.unsqueeze(0)  # (1, edu_num, edu_embedding)
    #         edu_encoded_x = edu_x_ if edu_encoded_x is None else torch.cat((edu_encoded_x, edu_x_), 0)
    #     edu_encoded_x = self.input_dense(edu_encoded_x)
    #     return edu_encoded_x, e_masks

    def encode_edus3(self, e_inputs, model_xl, tokenizer_xl):
        """ 实现对 EDU 的全局编码
        """
        e_txt, e_input_words, e_in_word_embeds, e_input_poses, e_masks = e_inputs
        # batch_size, max_edu_len, max_word_len = e_input_words.size()
        max_edu_len = len(e_txt[0])
        e_masks = (e_masks.sum(-1) > 0).int()
        e_x = torch.zeros([1, XLNET_SIZE]).float()  # pad 向量
        if USE_CUDA:
            e_x = e_x.cuda(CUDA_ID)
        edu_encoded_x = None
        for e_in_edus in e_txt:
            e_rep_idx = []
            # b_idx = 0
            b_idx = -1
            result_edu_s = []
            idx = 0
            while idx < len(e_in_edus):
                edu__ = e_in_edus[idx]
                encoded_ids = tokenizer_xl.encode(edu__, add_special_tokens=True)
                result_edu_s += encoded_ids
                b_idx += len(encoded_ids)
                e_rep_idx.append(b_idx)
                idx += 1
            result_edu_s = torch.tensor([result_edu_s])  # (1, 6)
            if USE_CUDA:
                result_edu_s = result_edu_s.cuda(CUDA_ID)
            txt_len = result_edu_s.size(1)
            k = (txt_len // CHUNK_SIZE) + 1
            outputs = None
            chunk_edu_ids = torch.chunk(result_edu_s, k, 1)
            for tmp_chunk in chunk_edu_ids:
                tmp_out = model_xl(tmp_chunk)[0].squeeze()
                outputs = tmp_out if outputs is None else torch.cat((outputs, tmp_out), 0)
            # outputs = model_xl(result_edu_s)[0].squeeze()
            if outputs is None:
                input("wrong")
            # input(outputs.size())
            e_rep_idx = torch.Tensor(e_rep_idx).long()
            if USE_CUDA:
                e_rep_idx = e_rep_idx.cuda(CUDA_ID)
            outputs = torch.index_select(outputs, 0, e_rep_idx)
            while idx < max_edu_len:
                idx += 1
                outputs = torch.cat((outputs, e_x), 0)
            edu_x_ = outputs.unsqueeze(0)  # (1, edu_num, edu_embedding)
            edu_encoded_x = edu_x_ if edu_encoded_x is None else torch.cat((edu_encoded_x, edu_x_), 0)
        edu_encoded_x = self.input_dense(edu_encoded_x)
        return edu_encoded_x, e_masks

    def decode_batch(self, e_outputs, e_contexts, d_in, d_masks):
        """ e_outputs: (batch, split_num, hidden)
            e_contexts: 对切割点的 rnn 表征上下文环境 (batch, hidden)
            解码输入，给出当前区域端点（d_in, d_masks）
        """
        # 得到输入中每个 edu 的每一次分割的 mask，以及数字形式的边界下标 d_in
        # (batch, split_num, 2), (batch, split_num, max_edu_len)
        batch_size = d_in.size()[0]
        # 对 d_masks 的 edu 维度 mask 值相加，0部分对应edu即为0 (batch, split_num)
        d_outputs_masks = (d_masks.sum(-1) > 0).type_as(d_masks)
        # 上下文编码, 初始状态 (batch, hidden)
        # d_init_states = self.context_dense(e_contexts)
        d_init_states = e_contexts

        # (2, split_num, batch_size, hidden) 取每个分割点两端的 context hidden 作为解码输入
        # e_outputs:(10, 114, 256)
        # torch.arange(batch_size): [[0,...,9]
        # d_in_: [2, 112, 10]] (2, split_num, batch)
        d_in_ = d_in.permute(2, 1, 0)
        d_inputs = e_outputs[torch.arange(batch_size), d_in_]
        # input(d_inputs.size())
        # (batch_size, split_num, 2, hidden)
        d_inputs = d_inputs.permute(2, 1, 0, 3)
        # (batch_size, split_num, 2 * hidden) 将边界信息压缩
        d_inputs = d_inputs.contiguous().view(d_inputs.size(0), d_inputs.size(1), -1)

        # mask 之后的输入 (batch, split_num, 2 * hidden)
        d_inputs = d_inputs * d_outputs_masks.unsqueeze(-1).float()

        # (batch, split_num, hidden)
        d_outputs = self.decoder.run_batch(d_inputs, d_init_states, d_outputs_masks, d_in)

        # 输出 split_num 个得分，根据若干得分计算下标
        return d_outputs, d_outputs_masks

    def loss(self, e_inputs, d_inputs, grounds, n_epoch=0, model_xl=None, tokenizer_xl=None):
        """ losses computation.
            结构预测参考基本编码解码，
            关系预测对分割点表征强化为内部节点，对重新编码的结果进行打分
        """
        split_ground, nr_ground, d_g_s, d_g_nr, l_x = grounds
        # (batch, max_edu_num, hidden), (batch, max_edu_num)
        e_txt, e_in_words, e_in_word_embeds, e_in_poses, e_masks, bound_info = e_inputs
        # e_in_words, e_in_word_embeds, e_in_poses, e_masks, bound_info = e_inputs
        e_h, e_masks = self.encode_edus((e_txt, e_in_words, e_in_word_embeds, e_in_poses, e_masks),
                                        model_xl, tokenizer_xl)
        # Encoder TD & BU
        if SPLIT_V == 0:
            splits, e_outputs_masks, lengths = self.context_encoder.get_splits(e_h, e_masks)
        elif SPLIT_V == 1:
            splits, e_outputs_masks, lengths = self.context_encoder.sim_split1(e_h, e_masks)
        else:
            splits, e_outputs_masks, lengths = self.context_encoder.sim_split2(e_h, e_masks)

        # splits embedded with boundaries
        if USE_BOUND:
            bound_embed = self.bound_emb(bound_info)
            splits_bounded = torch.cat((splits, bound_embed), -1)
            e_out, e_ctx = self.context_encoder(splits_bounded, lengths, e_outputs_masks)
        else:
            e_out, e_ctx = self.context_encoder(splits, lengths, e_outputs_masks)

        # Decoder
        d_in, d_masks = d_inputs
        d_out, d_out_masks = self.decode_batch(e_out, e_ctx, d_in, d_masks)

        # Span
        splits_attn = self.split_attention(e_out, d_out, d_masks)
        splits_predict_ = splits_attn.log_softmax(dim=2)  # 对 decoder 结果打分, (batch, split_num, split_num)
        split_ground_ = split_ground.view(-1)  # (batch, split_num)
        splits_predict = splits_predict_.view(split_ground_.size(0), -1)
        splits_masks = d_out_masks.view(-1).float()
        splits_loss = func.nll_loss(splits_predict, split_ground_, reduction="none")
        splits_loss = (splits_loss * splits_masks).sum() / splits_masks.sum()

        # NR
        nr_score = self.nr_classifier(e_out, d_out)
        nr_score = nr_score.log_softmax(dim=-1) * d_masks.unsqueeze(-1).float()
        nr_score = nr_score.view(nr_score.size(0) * nr_score.size(1), nr_score.size(2), nr_score.size(3))
        target_nr_score = nr_score[torch.arange(nr_score.size(0)), split_ground_]
        target_nr_ground = nr_ground.view(-1)
        nr_loss = func.nll_loss(target_nr_score, target_nr_ground)

        if n_epoch > WARM_UP_EP:
            batch_, split_num, _ = splits_predict_.size()
            gen_g_s, gen_g_nr = d_g_s.clone(), d_g_nr.clone()
            if USE_CUDA:
                gen_g_s = gen_g_s.cuda(CUDA_ID)
                gen_g_nr = gen_g_nr.cuda(CUDA_ID)
            nr_s = self.nr_linear(target_nr_score)
            nr_idx_ = 0
            for t_idx in range(batch_):
                splits_ = splits_predict_[t_idx]  # (split_num, split_num)
                act_len = min(splits_.size(0), MAX_W)
                split_ground_t = split_ground[t_idx]
                l_x_ = l_x[t_idx]
                s_num = len(l_x_)
                nr_idx = nr_idx_
                nr_idx_ += split_num
                buffer_idx = []
                for idx in range(s_num):
                    if l_x_[idx] < MAX_H:
                        # S graph
                        if l_x_[idx] in buffer_idx:
                            gen_g_s[t_idx][l_x_[idx]][:act_len] = gen_g_s[t_idx][l_x_[idx]][:act_len] + splits_[idx][:act_len]
                        else:
                            buffer_idx.append(l_x_[idx])
                            gen_g_s[t_idx][l_x_[idx]][:act_len] = splits_[idx][:act_len]
                        if split_ground_t[idx] < MAX_W:
                            # NR graph
                            gen_g_nr[t_idx][l_x_[idx]][split_ground_t[idx]] = nr_s[nr_idx] * NR_LABEL_NUM
                    nr_idx += 1
            if RANDOM_MASK_LEARN and random.random() < RMR:
                g_mask = d_g_s + 1
                gen_g_s = gen_g_s * g_mask
            real_img = torch.cat((d_g_s.unsqueeze(1), d_g_nr.unsqueeze(1)), 1)
            gen_img = torch.cat((gen_g_s.unsqueeze(1), gen_g_nr.unsqueeze(1)), 1)
        else:
            gen_img = real_img = None
        return splits_loss, nr_loss, gen_img, real_img

    def parse_predict(self, session):
        left, right = session.stack[-1]
        d_input = torch.cat([session.memory[0, left], session.memory[0, right]]).view(1, -1)
        d_output = self.decoder(d_input, session.state, (left, right))
        mask = torch.zeros(1, 1, session.n, dtype=torch.uint8)
        mask[0, 0, left: right + 1] = 1
        if USE_CUDA:
            mask = mask.cuda(CUDA_ID)
        split_scores = self.split_attention(session.memory, d_output.unsqueeze(1), mask)
        split_scores = split_scores.softmax(dim=-1).squeeze(0).squeeze(0)
        split_scores = split_scores.cpu().detach().numpy()

        nr_score = self.nr_classifier(session.memory, d_output.unsqueeze(1)).softmax(dim=-1) * mask.unsqueeze(
            -1).float()
        nr_score = nr_score[0, 0].cpu().detach().numpy()
        return split_scores, nr_score, d_output, mask

    def forward(self, session):
        pass


class MaskedGRU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MaskedGRU, self).__init__()
        self.rnn = nn.GRU(batch_first=True, *args, **kwargs)
        self.hidden_size = self.rnn.hidden_size

    def forward(self, padded, lengths, initial_state=None):
        # [batch*edu]
        zero_mask = lengths != 0
        lengths[lengths == 0] += 1  # in case zero length instance
        _, indices = lengths.sort(descending=True)
        _, rev_indices = indices.sort()

        # [batch*edu, max_word_len, embedding]
        padded_sorted = padded[indices]
        lengths_sorted = lengths[indices]
        padded_packed = pack_padded_sequence(padded_sorted, lengths_sorted, batch_first=True)
        self.rnn.flatten_parameters()
        outputs_sorted_packed, hidden_sorted = self.rnn(padded_packed, initial_state)
        # [batch*edu, max_word_len, ]
        outputs_sorted, _ = pad_packed_sequence(outputs_sorted_packed, batch_first=True)
        # [batch*edu, max_word_len, output_size]
        outputs = outputs_sorted[rev_indices]
        # [batch*edu, output_size]
        hidden = hidden_sorted.transpose(1, 0).contiguous().view(outputs.size(0), -1)[rev_indices]

        outputs = outputs * zero_mask.view(-1, 1, 1).float()
        hidden = hidden * zero_mask.view(-1, 1).float()
        return outputs, hidden


class BiGRUEDUEncoder(nn.Module):

    def __init__(self, input_size):
        super(BiGRUEDUEncoder, self).__init__()
        self.input_size = input_size
        self.masked_gru = MaskedGRU(input_size, HIDDEN_SIZE // 2, bidirectional=True)
        self.token_scorer = nn.Linear(HIDDEN_SIZE, 1)
        self.dropout = nn.Dropout(DROP_OUT)
        self.sft = nn.Softmax(dim=-1)

    def forward(self, inputs, masks):
        """ inputs:  (batch, edu_len, word_len, emb)
            masks:  (batch, edu_len, word_len)
        """
        lengths = masks.sum(-1)  # 统计一批 EDUs 里面每个 EDU 的实际长度，一起放进 padded GRU
        outputs, hidden = self.masked_gru(inputs, lengths)
        token_score = self.token_scorer(outputs).squeeze(-1)
        token_score[masks == 0] = -1e8  # pad 工作的平滑处理
        token_score = self.sft(token_score) * masks.float()
        weighted_sum = (outputs * token_score.unsqueeze(-1)).sum(-2)  # 对每个edu自身做注意力分配，求和得到edu表征
        outputs = hidden + self.dropout(weighted_sum)  # 混合表征
        return outputs


class SplitAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size):
        super(SplitAttention, self).__init__()
        self.bi_affine = BiAffineAttention(encoder_size, decoder_size, TRAN_LABEL_NUM, SPLIT_MLP_SIZE)

    def forward(self, e_outputs, d_outputs, masks):
        bi_affine = self.bi_affine(e_outputs, d_outputs)
        attn = bi_affine.squeeze(-1)
        attn[masks == 0] = -1e8
        return attn


class BiAffineAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size, num_labels, hidden_size):
        super(BiAffineAttention, self).__init__()
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.e_mlp = nn.Sequential(
            nn.Linear(encoder_size, hidden_size),
            nn.ReLU()
        )
        self.d_mlp = nn.Sequential(
            nn.Linear(decoder_size, hidden_size),
            nn.ReLU()
        )
        self.W_e = nn.Parameter(torch.empty(num_labels, hidden_size, dtype=torch.float))
        self.W_d = nn.Parameter(torch.empty(num_labels, hidden_size, dtype=torch.float))
        self.U = nn.Parameter(torch.empty(num_labels, hidden_size, hidden_size, dtype=torch.float))
        self.b = nn.Parameter(torch.zeros(num_labels, 1, 1, dtype=torch.float))
        nn.init.xavier_normal_(self.W_e)
        nn.init.xavier_normal_(self.W_d)
        nn.init.xavier_normal_(self.U)

    def forward(self, e_outputs, d_outputs):
        """ e_outputs: (batch, length_encoder, encoder_size)
            d_outputs: (batch, length_decoder, decoder_size)
            length_decoder = split_num
            length_encoder = split_num + 2
        """
        e_outputs = self.e_mlp(e_outputs)  # (batch, length_encoder, hidden)
        d_outputs = self.d_mlp(d_outputs)  # (batch, length_decoder, hidden)

        # 对每个分割点的 h 转换到标签空间（标签个数对应转换的类别数）
        out_e = (self.W_e @ e_outputs.transpose(1, 2)).unsqueeze(2)  # (batch, num_labels, 1, length_encoder)
        out_d = (self.W_d @ d_outputs.transpose(1, 2)).unsqueeze(3)  # (batch, num_labels, length_decoder, 1)

        # [batch, 1, length_decoder, hidden_size] @ [num_labels, hidden_size, hidden_size]
        # [batch, num_labels, length_decoder, hidden_size]
        out_u = d_outputs.unsqueeze(1) @ self.U

        # [batch, num_labels, length_decoder, hidden] @ [batch, 1, hidden, length_encoder]
        # [batch, num_labels, length_decoder, length_encoder]
        out_u = out_u @ e_outputs.unsqueeze(1).transpose(2, 3)

        # [batch, length_decoder, length_encoder, num_labels]
        out = (out_e + out_u + self.b).permute(0, 2, 3, 1) if OPT_ATTN else \
            (out_e + out_d + out_u + self.b).permute(0, 2, 3, 1)
        return out


class Context_Encoder(nn.Module):
    def __init__(self, input_size):
        super(Context_Encoder, self).__init__()
        self.input_size = input_size
        self.input_dense = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.edu_rnn = MaskedGRU(HIDDEN_SIZE, HIDDEN_SIZE // 2, bidirectional=True)
        self.edu_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.dropout = nn.Dropout(DROP_OUT)
        self.conv = nn.Sequential(
            nn.Conv1d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=KERNEL_SIZE, bias=False, stride=1),
            nn.ReLU(),
            nn.Dropout(DROP_OUT)
        )
        self.split_rnn = MaskedGRU(input_size, HIDDEN_SIZE // 2, bidirectional=True)
        self.split_rnn_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.output_size = HIDDEN_SIZE
        self.attn = MultiHeadAttention(heads=HEADS, input_size=HIDDEN_SIZE, hidden_size=ML_ATT_HIDDEN, dropout=DROP_OUT)
        # 加入self-attn
        self.edu_attn = MultiHeadAttention(heads=HEADS_e, input_size=HIDDEN_SIZE, hidden_size=ML_ATT_HIDDEN_e,
                                           dropout=DROP_OUT)
        self.attn_norm = nn.LayerNorm(HIDDEN_SIZE)

    def get_splits(self, edus_h, masks):
        """ edus_h: (batch, max_edu_num, hidden), masks: (batch, max_edu_num)
        """
        edus_h = self.input_dense(edus_h)
        # 对 EDU 使用 RNN 编码上下文信息并用残差连接
        edus, _ = self.edu_rnn(edus_h, masks.sum(-1))  # (batch, num_edus, hidden_size)
        if EDU_ATT:
            attn_masks = masks.unsqueeze(1).expand(masks.size(0), masks.size(1), masks.size(1)) * masks.unsqueeze(-1)
            edus = self.edu_attn(edus_h, edus_h, edus_h, attn_masks)
        # masks = torch.cat([(masks.sum(-1, keepdim=True) > 0).type_as(masks), masks], dim=1)
        masks = masks[:, :-1]
        edus = edus_h + self.dropout(edus)
        if LAYER_NORM_USE:
            edus = self.edu_norm(edus)
        # 用 CNN 编码切割点 (batch, hidden, max_edu_num)
        edus = edus.transpose(-2, -1)
        splits = self.conv(edus).transpose(-2, -1)  # (batch, split_num, hidden) 这个过程对每个篇章添加边界并作分割点的表征
        lengths = masks.sum(-1)
        return splits, masks, lengths

    def sim_split1(self, edus_h, masks):
        edus = edus_h
        if LAYER_NORM_USE:
            edus = self.edu_norm(edus)
        # 用 CNN 编码切割点 (batch, hidden, max_edu_num)
        edus = edus.transpose(-2, -1)
        splits = self.conv(edus).transpose(-2, -1)  # (batch, split_num, hidden) 这个过程对每个篇章添加边界并作分割点的表征
        masks = masks[:, :-1]
        lengths = masks.sum(-1)
        return splits, masks, lengths

    def sim_split2(self, edus_h, masks):
        edus = edus_h
        masks = masks[:, :-1]
        if LAYER_NORM_USE:
            edus = self.edu_norm(edus)
        splits = edus[:, :-1, :]
        lengths = masks.sum(-1)
        return splits, masks, lengths

    def forward(self, splits, lengths, masks):
        """ edus_h: (batch, max_edu_num, hidden), masks: (batch, max_edu_num)
        """
        outputs, hidden = self.split_rnn(splits, lengths)
        # outputs = splits + self.dropout(outputs)  # (batch, split_num, hidden)
        if LAYER_NORM_USE:
            outputs = self.split_rnn_norm(outputs)
        if CONTEXT_ATT:
            attn_masks = masks.unsqueeze(1).expand(masks.size(0), masks.size(1), masks.size(1)) * masks.unsqueeze(-1)
            outputs = outputs + self.attn(outputs, outputs, outputs, attn_masks)
            if LAYER_NORM_USE:
                outputs = self.attn_norm(outputs)
        return outputs, hidden


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_k = hidden_size // heads
        self.h = heads
        self.q_linear = nn.Linear(input_size, hidden_size)
        self.v_linear = nn.Linear(input_size, hidden_size)
        self.k_linear = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.d_k * self.h, input_size)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).float()
            scores = scores.masked_fill(mask == 0, -1e8)
        scores = func.softmax(scores, dim=-1)
        if mask is not None:
            scores = scores * mask
        scores = self.dropout(scores)
        attn = scores @ v

        # concatenate heads and put through final linear layer
        concat = attn.transpose(1, 2).contiguous().view(bs, -1, self.d_k * self.h)
        output = self.out(concat)
        if mask is not None:
            mask = (mask.squeeze(1).sum(-1, keepdim=True) > 0).float()
            output = output * mask
        return output


class Decoder(nn.Module):
    def __init__(self, inputs_size):
        super(Decoder, self).__init__()
        self.rnn_cell = nn.GRUCell(inputs_size, HIDDEN_SIZE)
        self.output_size = HIDDEN_SIZE
        self.t_outputs = None
        # parse
        self.h_stack, self.s_stack = None, None
        self.left, self.right = 0, 0

    def init_stack(self, left, right):
        self.h_stack, self.s_stack = [], []
        self.left, self.right = left, right

    def forward(self, in_, h_state, d_idx):
        """ input_: 当前区域的边界向量信息 (batch_size, split_num, 2 * hidden)
            state: 当前区域的上下文状态 (batch_size=1, split_num=1, 2 * hidden)
            d_idx: 当前输入的边界下标
        """
        h_state = self.rnn_cell(in_, h_state)
        return h_state

    def decode_one(self, in_, h_state, d_idx):
        """ 解码一篇文章，算法详见 PPT DP2019——lyzhang
            input: (split_num, 2 * hidden), 这里的 split_num 指的是最大的 edu 个数减去1，对于超出的部分，(0, 0)
            d_idx: (split_num, 2)
            init_state: (2 * hidden)
            output: ?
        """
        outputs = None
        h_state = h_state.unsqueeze(0)
        thread_name = current_thread().getName()
        split_num = in_.size()[0]
        for i in range(split_num):
            decode_in = in_[i].unsqueeze(0)
            h_state = self.rnn_cell(decode_in, h_state)
            # 对 h_state 做解码输出的拼接
            outputs = h_state if outputs is None else torch.cat((outputs, h_state), 0)
        self.t_outputs[thread_name] = outputs

    def run_batch(self, inputs, init_states, masks, d_in):
        d_in = d_in.cpu().detach().numpy()
        inputs = inputs * masks.unsqueeze(-1).float()  # 将该是 0 的地方直接设置为 0
        thread_num = inputs.size()[0]
        self.t_outputs = dict()
        thread_list = []
        for idx in range(thread_num):
            arguments = (inputs[idx], init_states[idx], d_in[idx])
            t = Thread(target=self.decode_one, args=arguments, name="Thread_" + str(idx))
            thread_list.append(t)
        for t in thread_list:
            t.start()
        for t in thread_list:
            t.join()
        # 对输出作转换 (batch_size, split_num, hidden)
        outputs = None
        for idx in range(thread_num):
            key = "Thread_" + str(idx)
            output = self.t_outputs[key].unsqueeze(0)
            outputs = output if outputs is None else torch.cat((outputs, output), 0)
        outputs = outputs * masks.unsqueeze(-1).float()
        return outputs


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Up—sampling
        self.down = nn.Sequential(nn.Conv2d(in_channel_G, out_channel_G, (ker_h_G, ker_w_G), strip_G),
                                  nn.ReLU())
        # max pooling
        self.max_p = nn.MaxPool2d(kernel_size=(p_w_G, p_h_G), stride=p_w_G)

        # Fully-connected layers
        c_h = (MAX_H - (ker_h_G - 1)) // p_w_G if MAX_POOLING else MAX_H - (ker_h_G - 1)
        c_w = (MAX_W - (ker_w_G - 1)) // p_h_G if MAX_POOLING else MAX_W - (ker_w_G - 1)
        down_dim = out_channel_G * c_h * c_w
        self.fc = nn.Sequential(
            # nn.BatchNorm1d(down_dim, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(down_dim, 1),
            # nn.Sigmoid()
            nn.Linear(down_dim, down_dim // 2),
            # nn.BatchNorm1d(down_dim // 2, 0.8),
            nn.LayerNorm(down_dim // 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(down_dim // 2, 1),
        )

    def cnn_feat_ext(self, img):
        out = self.down(img)
        if MAX_POOLING:
            out = self.max_p(out)
        return out

    def forward(self, out):
        """ (batch, colors, height, width)
            (5, 3, 20, 80)
            16 * 19 * 1 = 304
        """
        # out = self.down(img)
        # if MAX_POOLING:
        #     out = self.max_p(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

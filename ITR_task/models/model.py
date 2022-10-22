from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F
from torch.autograd import *
from .modules import masked_mean, masked_max, Attention, CoAttention, MaskedSoftmax, MyMultiHeadAttention
from transformers import AutoTokenizer, BertModel, BertConfig, RobertaTokenizer, RobertaConfig, RobertaModel
from transformers import AutoModel, AutoTokenizer

class InformationFusion(nn.Module):
    def __init__(self, hidden_size):
        super(InformationFusion, self).__init__()
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
    def forward(self, query_hidden_state, tgt_hidden_state):
        merge_representation = torch.cat((query_hidden_state, tgt_hidden_state), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        gated_converted_hidden = torch.mul(gate_value, tgt_hidden_state)
        return gated_converted_hidden


class MultimodalEncoder(nn.Module):
    def __init__(self, opt):
        """Initialize model."""
        super(MultimodalEncoder, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.hidden_size = opt.hidden_size
        self.bert_hidden_size = opt.bert_hidden_size
        self.fc_feat_size = opt.fc_feat_size

        self.bert_layer_num = opt.bert_layer_num

        self.init_bert(self.bert_layer_num)
        self.dropout = nn.Dropout(p=opt.dropout)
        self.linear_classifer_final = nn.Linear(opt.bert_hidden_size, opt.trg_class)
        self.linear_img = nn.Linear(self.fc_feat_size, self.bert_hidden_size)

        # self.linear_text = nn.Linear(self.bert_hidden_size, self.bert_hidden_size)
        # self.linear_com = nn.Linear(self.bert_hidden_size, self.bert_hidden_size)
        self.linear_text = nn.Sequential(nn.Linear(self.bert_hidden_size, self.bert_hidden_size), nn.Dropout(p=opt.dropout))
        self.linear_com = nn.Sequential(nn.Linear(self.bert_hidden_size, self.bert_hidden_size), nn.Dropout(p=opt.dropout))



        # self.project_concat_img = nn.Linear(self.bert_hidden_size * 2, opt.bert_hidden_size)
        # self.project_concat_cap = nn.Linear(self.bert_hidden_size * 2, opt.bert_hidden_size)

        self.attention_img = Attention(self.bert_hidden_size, self.bert_hidden_size)
        self.com_num_use = opt.com_num_use

        self.attention_text = Attention(self.bert_hidden_size, self.bert_hidden_size)
        self.project_fusion = nn.Linear(self.bert_hidden_size * 2, opt.bert_hidden_size)
        # self.project_fusion_img_task = nn.Linear(self.bert_hidden_size * 3, opt.bert_hidden_size)
        # self.project_fusion_text_task = nn.Linear(self.bert_hidden_size * 3, opt.bert_hidden_size)
        self.project_fusion_img_text_task = nn.Linear(self.bert_hidden_size * 3, opt.bert_hidden_size)

        # self.info_gate = InformationFusion(self.bert_hidden_size)
    # def init_bert(self, layer_num):
    #     bert_version = "vinai/bertweet-base"
    #     self.tokenizer = AutoTokenizer.from_pretrained(bert_version)
    #     # since the framework of bertweet is same with  RoBERTa, so we directly use the RobertaConfig
    #     # reference: https://github.com/VinAIResearch/BERTweet/issues/17
    #     config = RobertaConfig.from_pretrained(bert_version)
    #     config.num_hidden_layers = layer_num
    #     self.bert_encoder = AutoModel.from_pretrained(bert_version, config=config).to(self.device)  # auto skip unused layers
    def init_bert(self, layer_num):
        # bert_version = "vinai/bertweet-base"
        bert_version = "bert-base-uncased"

        self.tokenizer = AutoTokenizer.from_pretrained(bert_version)
        # config = RobertaConfig.from_pretrained(bert_version)
        config = BertConfig.from_pretrained(bert_version)
        config.num_hidden_layers = layer_num
        self.bert_encoder = AutoModel.from_pretrained(bert_version, config=config).to(
            self.device)  # auto skip unused layers




    def get_text_feat(self, memory_bank, text_pooling_type='max', mask=None):
        # map memory bank into one feat vector using mask
        assert len(memory_bank.shape) == 3

        if text_pooling_type == 'max':
            text_feats = masked_max(memory_bank, mask, dim=1)
        elif text_pooling_type == 'avg':
            text_feats = masked_mean(memory_bank, mask, dim=1)
        return text_feats

    def encode_text(self, texts):
        texts_new = [text for text in texts]
        input = self.tokenizer(texts_new, padding=True, truncation=True, max_length=50,
                               return_tensors="pt").to(self.device)
        attention_mask = list(input.values())[-1]
        out_states = self.bert_encoder(**input)[0]  # [0] is the last_hidden_state, [1] is the pooled_output
        return out_states, attention_mask

    def encode_img(self, img_feats):
        # read image visual feature and map them to bi_hidden_size
        # img_feats: [batch, 2048] for resnet152
        batch_size = img_feats.shape[0]
        img_feats = img_feats.view(-1, img_feats.shape[1])
        img_feats = self.linear_img(img_feats)
        # img_feats = img_feats.view(batch_size, -1, img_feats.shape[-1])  # [batch_size, 49, bert_hidden_size]
        return img_feats

    def forward(self, fc_feats, texts, ret_coms):
        # ret_coms: [batch* opt.com_num_use]
        img_feats = self.encode_img(fc_feats)
        text_states, text_masks = self.encode_text(texts)
        enc_text = self.get_text_feat(text_states, text_pooling_type='max', mask=text_masks)
        enc_text = self.linear_text(enc_text)
        # img2text_attend = self.img2text_att(img_feats, enc_text, text_masks)

        combined_feat = torch.cat((enc_text, img_feats), dim=1)
        fusion_vec = self.project_fusion(combined_feat)

        if  not self.opt.use_com:

            classifier_outputs = self.linear_classifer_final(fusion_vec)




        else:
            com_states, com_masks = self.encode_text(ret_coms)
            enc_com = self.get_text_feat(com_states, text_pooling_type='max', mask=com_masks)
            enc_com = self.linear_com(enc_com)  # [batch* opt.com_num_use, bert_hidden_size]
            batch_size = img_feats.size(0)
            enc_com = enc_com.contiguous().view(batch_size, -1, self.bert_hidden_size)

            text2com = self.attention_text(enc_text, enc_com)
            img2com = self.attention_img(img_feats, enc_com)
            combined_feat = torch.cat((fusion_vec, text2com, img2com), dim=1)
            proj_feat = self.project_fusion_img_text_task(combined_feat)
            classifier_outputs = self.linear_classifer_final(proj_feat)


            # gate_com = self.info_gate(fusion_vec, attended_com)


        return classifier_outputs


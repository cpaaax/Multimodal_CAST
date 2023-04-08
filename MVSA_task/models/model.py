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
        self.linear_classifer_final =nn.Linear(opt.hidden_size, opt.tgt_class)
        self.linear_fc_img = nn.Linear(self.fc_feat_size, self.hidden_size)
        self.linear_att_img = nn.Linear(self.fc_feat_size, self.hidden_size)


        self.linear_fc_text = nn.Linear(self.bert_hidden_size, self.hidden_size)
        self.linear_att_text = nn.Linear(self.bert_hidden_size, self.hidden_size)

        self.linear_com = nn.Linear(self.bert_hidden_size, self.hidden_size)

        # self.linear_text = nn.Linear(self.bert_hidden_size, self.bert_hidden_size)
        # self.linear_com = nn.Linear(self.bert_hidden_size, self.bert_hidden_size)





        self.com_num_use = opt.com_num_use

        self.project_fusion = nn.Linear(self.hidden_size * 2, opt.hidden_size)
        self.project_fusion_com = nn.Linear(self.hidden_size * 2, opt.hidden_size)
        self.fusion_com_linear = nn.Linear(self.hidden_size * 2, opt.hidden_size)





        # for the first hop
        self.att_img_1 = Attention(self.hidden_size)
        self.att_text_1 = Attention(self.hidden_size)
        # for other hop
        self.img_att_img2text_other = Attention(self.hidden_size)
        self.img_att_text2img_other = Attention(self.hidden_size)

        self.text_att_text2img_other = Attention(self.hidden_size)
        self.text_att_img2text_other = Attention(self.hidden_size)

        self.attention_fusion_img = Attention(self.hidden_size)
        self.attention_fusion_text = Attention(self.hidden_size)
        # self.project_img2com = nn.Linear(self.hidden_size, opt.hidden_size)
        # self.project_text2com = nn.Linear(self.hidden_size, opt.hidden_size)

    def init_bert(self, layer_num):
        bert_version = "bert-base-uncased"

        self.tokenizer = AutoTokenizer.from_pretrained(bert_version)
        # since the framework of bertweet is same with  RoBERTa, so we directly use the RobertaConfig
        # reference: https://github.com/VinAIResearch/BERTweet/issues/17
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
        texts = [i for i in texts]
        input = self.tokenizer(texts, padding=True, truncation=True, max_length=50,
                               return_tensors="pt").to(self.device)
        attention_mask = list(input.values())[-1]
        out_states = self.bert_encoder(**input)[0]  # [0] is the last_hidden_state, [1] is the pooled_output
        return out_states, attention_mask

    def encode_img(self, img_feats):
        # read image visual feature and map them to bi_hidden_size
        # img_feats: [batch, 49, 2048] for resnet152
        batch_size = img_feats.shape[0]

        img_feats = masked_mean(img_feats, dim=1)
        fc_feats = self.linear_fc_img(img_feats)
        # img_feats = img_feats.view(batch_size, -1, img_feats.shape[-1])  # [batch_size, 49, bert_hidden_size]
        return fc_feats

    def forward(self, feats, texts, ret_coms):
        # ret_coms: [batch* opt.com_num_use]
        # feats: [batch, 7,7,2048], texts: [batch]
        text_states, text_masks = self.encode_text(texts)
        fc_enc_text = self.get_text_feat(text_states, text_pooling_type='max', mask=text_masks)
        fc_enc_text = self.linear_fc_text(fc_enc_text)
        att_enc_text = self.linear_att_text(text_states)

        batch = feats.size(0)
        img_feat_size = feats.size(-1)
        feats = feats.view(batch, -1, img_feat_size)
        fc_img_feats = self.encode_img(feats)
        att_img_feats = self.linear_att_img(feats)

        hop_img_1 = self.att_img_1(fc_img_feats, att_img_feats)
        hop_text_1 = self.att_text_1(fc_enc_text, att_enc_text, text_masks)

        # for the remain 5 hops attention of img
        hop_img_2 = self.img_att_img2text_other(hop_img_1, att_enc_text, text_masks)
        hop_img_3 = self.img_att_text2img_other(hop_img_2, att_img_feats)
        hop_img_4 = self.img_att_img2text_other(hop_img_3, att_enc_text, text_masks)
        hop_img_5 = self.img_att_text2img_other(hop_img_4, att_img_feats)

        # for the remain 5 hops attention of text
        hop_text_2 = self.text_att_text2img_other(hop_text_1, att_img_feats)
        hop_text_3 = self.text_att_img2text_other(hop_text_2, att_enc_text, text_masks)
        hop_text_4 = self.text_att_text2img_other(hop_text_3, att_img_feats)
        hop_text_5 = self.text_att_img2text_other(hop_text_4, att_enc_text, text_masks)




        combined_feat = torch.cat((hop_img_5, hop_text_5), dim=1)
        fusion_vec = self.project_fusion(combined_feat)

        if not self.opt.use_com:
            classifier_outputs = self.linear_classifer_final(fusion_vec)
        else:
            com_states, com_masks = self.encode_text(ret_coms)
            enc_com = self.get_text_feat(com_states, text_pooling_type='max', mask=com_masks)
            enc_com = self.linear_com(enc_com)  # [batch* opt.com_num_use, bert_hidden_size]
            batch_size = att_img_feats.size(0)
            enc_com = enc_com.contiguous().view(batch_size, -1, self.hidden_size)

            fusion_img2com = self.attention_fusion_img(fc_img_feats, enc_com)
            fusion_text2com = self.attention_fusion_text(fc_enc_text, enc_com)
            # fusion_img2com = self.project_img2com(fusion_img2com)
            # fusion_text2com = self.project_text2com(fusion_text2com)
            fusion_com = self.fusion_com_linear(torch.cat((fusion_img2com, fusion_text2com), dim=1))

            combined_feat = torch.cat((fusion_com, fusion_vec), dim=1)

            proj_feat = self.project_fusion_com(combined_feat)
            classifier_outputs = self.linear_classifer_final(proj_feat)
            # text2com = self.attention_text(enc_text, enc_com)
            # img2com = self.attention_img(fc_img_feats, enc_com)
            # combined_feat = torch.cat((enc_text, fc_img_feats, text2com, img2com), dim=1)
            # proj_feat = self.project_fusion_img_text_task(combined_feat)
            # classifier_outputs = self.linear_classifer_final(proj_feat)

        return classifier_outputs


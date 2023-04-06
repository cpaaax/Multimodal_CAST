import json

import torch

from .ModalityFusion import ModalityFusion
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from .modules import masked_mean, masked_max, Attention, CoAttention, MaskedSoftmax
from transformers import AutoTokenizer, BertModel, BertConfig, RobertaTokenizer, RobertaConfig, RobertaModel
from transformers import AutoModel, AutoTokenizer



class ClassificationLayer(torch.nn.Module):
    def __init__(self, dropout_rate=0):
        super(ClassificationLayer, self).__init__()
        self.Linear_1 = torch.nn.Linear(512, 256)
        self.Linear_2 = torch.nn.Linear(256, 2)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, input):
        hidden = self.Linear_1(input)
        hidden = self.dropout(hidden)

        # output = torch.softmax(self.Linear_2(hidden), dim=1)
        output = self.Linear_2(hidden)

        return output

class SelfTrainLoss(nn.Module):
    def __init__(self):
        super(SelfTrainLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean',log_target =False)
    def forward(self, input, target, mask=None):
        # input:[batch, len, label_num], target:[batch, len, label_num] mask:[batch, len]
        input = F.log_softmax(input, dim=1)
        target = F.softmax(target, dim=1)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            input_with_mask = input*mask
            targe_with_mask = target*mask
            loss = self.criterion(input_with_mask, targe_with_mask)
        else:
            loss = self.criterion(input, target)
        return loss


class ExtractImageFeature(torch.nn.Module):
    def __init__(self):
        super(ExtractImageFeature, self).__init__()
        # 2048->1024
        self.Linear = torch.nn.Linear(2048, 1024)

    def forward(self, input):
        batch_size = input.size(0)
        dimension = input.size(3)

        # input=input.reshape(batch_size,-1, dimension).permute(1,0,2)
        # output=list()
        # for i in range(196):
        #     sub_output=torch.nn.functional.relu(self.Linear(input[i]))
        #     output.append(sub_output)
        # output=torch.stack(output)
        # mean=torch.mean(output,0)
        # return mean,output
        input=input.reshape(batch_size,-1, dimension)  #[batch,196,2048]

        output = self.Linear(input)  #[batch,196,1024]
        mean = torch.mean(output,1) #[batch,1024]
        return mean,output







class Multimodel(torch.nn.Module):
    def __init__(self, device, opt):
        super(Multimodel, self).__init__()
        self.device = device
        self.opt = opt
        self.fuse = ModalityFusion(opt)


        # encode img
        self.img_linear = torch.nn.Linear(2048, opt.hidden_size)
        self.text_linear = torch.nn.Linear(opt.bert_hidden_size, opt.hidden_size)
        self.attr_linear = torch.nn.Linear(opt.bert_hidden_size, opt.hidden_size)
        self.com_linear = torch.nn.Linear(opt.bert_hidden_size, opt.hidden_size)

        self.text_linear_fc = torch.nn.Linear(opt.bert_hidden_size, opt.hidden_size)
        self.attr_linear_fc = torch.nn.Linear(opt.bert_hidden_size, opt.hidden_size)
        self.com_linear_fc = torch.nn.Linear(opt.bert_hidden_size, opt.hidden_size)

        self.bert_layer_num = opt.bert_layer_num
        self.init_bert(self.bert_layer_num)


        self.final_classifier = nn.Sequential(torch.nn.Linear(opt.hidden_size, opt.hidden_size), torch.nn.Dropout(opt.dropout),
                                              torch.nn.Linear(opt.hidden_size, opt.tgt_class))



    def init_bert(self, layer_num):
        # bert_version = "vinai/bertweet-base"
        bert_version = "bert-base-uncased"

        self.tokenizer = AutoTokenizer.from_pretrained(bert_version)
        # since the framework of bertweet is same with  RoBERTa, so we directly use the RobertaConfig
        # reference: https://github.com/VinAIResearch/BERTweet/issues/17
        # config = RobertaConfig.from_pretrained(bert_version)
        config = BertConfig.from_pretrained(bert_version)
        config.num_hidden_layers = layer_num
        self.bert_encoder = AutoModel.from_pretrained(bert_version, config=config).to(
            self.device)  # auto skip unused layers

    def encode_img(self, img):
        batch_size = img.size(0)
        dimension = img.size(3)

        input = img.contiguous().view(batch_size, -1, dimension)  # [batch,196,2048]
        att = self.img_linear(input)
        fc = torch.mean(att, 1)  # [batch,bert_hidden_size]

        return fc, att


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




    def forward(self, text, image_feature, attribute, com):
        image_fc, image_att = self.encode_img(image_feature)

        attr_states, attr_masks = self.encode_text(attribute)
        attr_fc = self.get_text_feat(attr_states, text_pooling_type='max', mask=attr_masks)
        attr_fc = self.attr_linear_fc(attr_fc)
        attr_states = self.attr_linear(attr_states)

        text_states, text_masks = self.encode_text(text)
        text_fc = self.get_text_feat(text_states, text_pooling_type='max', mask=text_masks)
        text_fc = self.text_linear_fc(text_fc)
        text_states = self.text_linear(text_states)


        com_states, com_masks = self.encode_text(com)
        com_fc = self.get_text_feat(com_states, text_pooling_type='max', mask=com_masks)
        com_fc = self.com_linear_fc(com_fc)


        batch_size = image_fc.size(0)
        com_fc = com_fc.contiguous().view(batch_size, -1, self.opt.hidden_size)

        # com_index:[batch, opt.com_num_use,30]->[batch*opt.com_num_use,30]
        fusion = self.fuse(image_fc, image_att, text_fc, text_states, attr_fc,
                           attr_states, com_fc)
        output = self.final_classifier(fusion)
        return output

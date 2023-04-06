import torch
import LoadData

import torch.nn as nn
import torch.nn.functional as F



class Attention(nn.Module):
    def __init__(self, hidden_size, feature_size):
        super(Attention, self).__init__()

        self.attn = nn.Linear(hidden_size+feature_size, int((hidden_size+feature_size)/2))
        self.linear_out = nn.Linear(int((hidden_size+feature_size)/2),1)
        self.softmax = nn.Softmax()

    def forward(self, hidden, encoder_outputs, encoder_mask=None):
        '''
        :param hidden: (batch_size, hidden_size)  一个step计算一次copy_attention
        :param encoder_outputs: (batch_size, len, rnn_size)
        :return:
            attn_energies  (batch_size*seq_per_img, 1, similar_len): the attention energies before softmax
        '''

        att_size = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, att_size, 1)  # (batch_size, 1, hidden_size)
        cat_feature = torch.cat([hidden, encoder_outputs], dim=2) # (batch_size, att_size, hidden_size+feature_size)
        cat_feature = self.attn(cat_feature) # (batch_size, att_size, (hidden_size+feature_size)/2)
        energies = torch.tanh(self.linear_out(cat_feature)).squeeze() # (batch_size, att_size)
        weight = F.softmax(energies, dim=1)  # (batch_size, att_size)

        if encoder_mask is not None:
            att_masks = encoder_mask[:, :att_size]

            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = encoder_outputs.view(-1, att_size, encoder_outputs.size(-1))  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
        return weight, att_res

class RepresentationFusion_revise(torch.nn.Module):
    def __init__(self, qv_feature_size, query1_feature_size, query2_feature_size):
        super(RepresentationFusion_revise, self).__init__()
        self.qv_to_qv_att = Attention(qv_feature_size, qv_feature_size)
        self.qv_to_q1_att = Attention(query1_feature_size, qv_feature_size)
        self.qv_to_q2_att = Attention(query2_feature_size, qv_feature_size)
    def forward(self, qv_feature, query1_feature, query2_feature, qv_att_feature):
        qv_to_qv_att_weight,_ = self.qv_to_qv_att(qv_feature, qv_att_feature)
        qv_to_q1_att_weight,_ = self.qv_to_q1_att(query1_feature, qv_att_feature)
        qv_to_q2_att_weight,_ = self.qv_to_q2_att(query2_feature, qv_att_feature)
        out = torch.mean((qv_to_qv_att_weight+qv_to_q1_att_weight+qv_to_q2_att_weight).unsqueeze(2)*qv_att_feature/3,1)
        return out


class ModalityFusion(torch.nn.Module):
    def __init__(self, opt):
        super(ModalityFusion, self).__init__()
        self.opt = opt
        image_feature_size=opt.hidden_size#image_feature.size(1)
        text_feature_size=opt.hidden_size#text_feature.size(1)
        attribute_feature_size=opt.hidden_size#attribute_feature.size(1)


        self.image_attention = RepresentationFusion_revise(image_feature_size, text_feature_size, attribute_feature_size)
        self.text_attention = RepresentationFusion_revise(text_feature_size, image_feature_size, attribute_feature_size)
        self.attribute_attention = RepresentationFusion_revise(attribute_feature_size, image_feature_size, text_feature_size)
        # self.com_attention = RepresentationFusion_revise(text_feature_size, image_feature_size,
        #                                                  text_feature_size)

        self.image_linear_1=torch.nn.Linear(image_feature_size,opt.hidden_size)
        self.text_linear_1=torch.nn.Linear(text_feature_size,opt.hidden_size)
        self.attribute_linear_1=torch.nn.Linear(attribute_feature_size,opt.hidden_size)
        self.image_linear_2=torch.nn.Linear(opt.hidden_size,1)
        self.text_linear_2=torch.nn.Linear(opt.hidden_size,1)
        self.attribute_linear_2=torch.nn.Linear(opt.hidden_size,1)
        self.image_linear_3=torch.nn.Linear(image_feature_size,opt.hidden_size)
        self.text_linear_3=torch.nn.Linear(text_feature_size,opt.hidden_size)
        self.attribute_linear_3=torch.nn.Linear(attribute_feature_size,opt.hidden_size)

        # self.fusion2com_attention = Attention(opt.hidden_size, opt.hidden_size)
        #
        # self.com_linear_1 = torch.nn.Linear(text_feature_size, opt.hidden_size)
        # self.com_linear_2 = torch.nn.Linear(opt.hidden_size, 1)
        # self.com_linear_3 = torch.nn.Linear(text_feature_size, opt.hidden_size)
        self.com_img = torch.nn.Linear(text_feature_size, opt.hidden_size)
        self.com_text = torch.nn.Linear(text_feature_size, opt.hidden_size)
        self.img2com_attention = Attention(opt.hidden_size, opt.hidden_size)
        self.text2com_attention = Attention(opt.hidden_size, opt.hidden_size)
        self.com_img_att_linear = torch.nn.Linear(text_feature_size, opt.hidden_size)
        self.com_text_att_linear = torch.nn.Linear(text_feature_size, opt.hidden_size)
        self.com_out_linear = torch.nn.Linear(opt.hidden_size*2, opt.hidden_size)
        self.com_fusion_linear = torch.nn.Linear(opt.hidden_size*2, opt.hidden_size)


    def forward(self, image_feature,image_seq,text_feature,text_seq,attribute_feature,attribute_seq, com_feature):
        # image_feature: [batch, opt.bert_hidden_size]           image_seq: [batch,196,opt.bert_hidden_size]
        # text_feature: [batch, opt.bert_hidden_size]             text_seq: [batch,30,opt.bert_hidden_size]
        # attribute_feature: [batch, opt.bert_hidden_size]        attribute_seq: [batch, 5, opt.bert_hidden_size]
        # com_feature: [batch,com_num_use,opt.bert_hidden_size]   com_seq: [batch,com_num_use,length,opt.bert_hidden_size]
        image_vector = self.image_attention(image_feature,text_feature,attribute_feature,image_seq)

        text_vector = self.text_attention(text_feature,image_feature,attribute_feature,text_seq)

        attribute_vector = self.attribute_attention(attribute_feature,image_feature,text_feature,attribute_seq)

        # # ****************
        # com_vector = self.com_attention(torch.mean(com_feature,dim=1),image_feature,text_feature,com_feature)


        image_hidden = torch.tanh(self.image_linear_1(image_vector))
        text_hidden = torch.tanh(self.text_linear_1(text_vector))
        attribute_hidden = torch.tanh(self.attribute_linear_1(attribute_vector))

        # ***************
        # com_hidden = torch.tanh(self.com_linear_1(com_vector))
        # ***************

        image_score=self.image_linear_2(image_hidden)
        text_score=self.text_linear_2(text_hidden)
        attribute_score=self.attribute_linear_2(attribute_hidden)

        #********
        # com_score = self.com_linear_2(com_hidden)
        #********


        score=torch.nn.functional.softmax(torch.cat([image_score,text_score,attribute_score],dim=1),dim=1)  # [batch,3]
        image_vector=self.image_linear_3(image_vector)
        text_vector=self.text_linear_3(text_vector)
        attribute_vector=self.attribute_linear_3(attribute_vector)

        #***************
        # com_vector = torch.tanh(self.com_linear_3(com_vector))
        last_vector = torch.cat(
            [image_vector.unsqueeze(1), text_vector.unsqueeze(1), attribute_vector.unsqueeze(1)],
            dim=1)  # [batch,3, 512]
        # final fuse
        last_fusion_vec = torch.bmm(score.unsqueeze(1), last_vector).squeeze()

        if self.opt.use_com:

            com_hidden_img = self.com_img(com_feature)
            com_hidden_text = self.com_text(com_feature)

            _, img2com_out = self.img2com_attention(image_vector, com_hidden_img)
            img2com_out = self.com_img_att_linear(img2com_out)

            _, text2com_out = self.text2com_attention(text_vector, com_hidden_text)
            text2com_out = self.com_text_att_linear(text2com_out)
            com_fusion = self.com_fusion_linear(torch.cat([text2com_out, img2com_out], dim=1))


            com_output = torch.cat([last_fusion_vec, com_fusion], dim=1)
            output = self.com_out_linear(com_output)

        else:

            output = last_fusion_vec
        return output

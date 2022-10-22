import torch
import torch.nn as nn
import torch.nn.functional as F
from .multi_head_att.submodules import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding


def masked_mean(input, mask=None, dim=1):
    # input: [batch_size, seq_len, hidden_size]
    # mask: Float Tensor of size [batch_size, seq_len], where 1.0 for unmask, 0.0 for mask ones
    if mask is None:
        return torch.mean(input, dim=dim)
    else:
        length = input.size(1)
        mask = mask[:,:length].unsqueeze(-1)
        mask_input = input * mask
        sum_mask_input = mask_input.sum(dim=dim)
        mask_ = mask.sum(dim=dim)
        sum_mask_out = sum_mask_input/mask_
        return sum_mask_out


def masked_max(input, mask=None, dim=1):
    # input: [batch_size, seq_len, hidden_size]
    # mask: Float Tensor of size [batch_size, seq_len], where 1.0 for unmask, 0.0 for mask ones
    if mask is None:
        max_v, _ = torch.max(input, dim=dim)
        return max_v
    else:
        length = input.size(1)
        mask = mask[:,:length].unsqueeze(-1)
        mask = mask.repeat(1, 1, input.size(-1))
        input = input.masked_fill(mask == 0.0, float('-inf'))
        max_v, _ = torch.max(input, dim=dim)
        return max_v


class MaskedSoftmax(nn.Module):
    def __init__(self, dim):
        super(MaskedSoftmax, self).__init__()
        self.dim = dim

    def forward(self, logit, mask=None):
        if mask is None:
            dist = F.softmax(logit - torch.max(logit, dim=self.dim, keepdim=True)[0], dim=self.dim)
        else:
            dist_ = F.softmax(logit - torch.max(logit, dim=self.dim, keepdim=True)[0], dim=self.dim) * mask
            normalization_factor = dist_.sum(self.dim, keepdim=True)
            dist = dist_ / normalization_factor
        return dist


# class Attention(nn.Module):
#     def __init__(self, rnn_size, hidden_size):
#         super(Attention, self).__init__()
#         self.rnn_size =rnn_size
#         self.att_hid_size = hidden_size
#         self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
#         self.alpha_net = nn.Linear(self.att_hid_size, 1)
#
#     def forward(self, h, att_feats, p_att_feats, att_masks=None):
#
#         if att_masks is not None:
#             length = att_feats.size(1)
#             att_masks = att_masks[:, :length]
#
#         # The p_att_feats here is already projected
#         att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
#         att = p_att_feats.view(-1, att_size, self.att_hid_size)
#
#         att_h = self.h2att(h)  # batch * att_hid_size
#         att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
#         dot = att + att_h  # batch * att_size * att_hid_size
#         dot = F.tanh(dot)  # batch * att_size * att_hid_size
#         dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
#         dot = self.alpha_net(dot)  # (batch * att_size) * 1
#         dot = dot.view(-1, att_size)  # batch * att_size
#
#         weight = F.softmax(dot, dim=1)  # batch * att_size
#         if att_masks is not None:
#             weight = weight * att_masks.view(-1, att_size).float()
#             weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
#         att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))  # batch * att_size * att_feat_size
#         att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
#
#         return att_res

class Attention(nn.Module):
    def __init__(self, rnn_size, hidden_size):
        super(Attention, self).__init__()

        self.attn = nn.Linear(rnn_size, hidden_size)
        self.input_project = nn.Linear(rnn_size, hidden_size)
        self.softmax = nn.Softmax()

    def forward(self, hidden, encoder_outputs, encoder_mask=None):
        '''
        :param hidden: (batch_size, hidden_size)  一个step计算一次copy_attention
        :param encoder_outputs: (batch_size, len, rnn_size)
        :return:
            attn_energies  (batch_size*seq_per_img, 1, similar_len): the attention energies before softmax
        '''
        hidden = self.input_project(hidden)
        att_size = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)
        energies = self.attn(encoder_outputs)  # (batch, len, hidden_size)
        attn_energies = torch.bmm(hidden, energies.transpose(1, 2))  # (batch, 1, len)
        attn_energies = attn_energies.view(-1, att_size)  # (batch, len)
        weight = F.softmax(attn_energies, dim=1)  # (batch, att_size)

        if encoder_mask is not None:
            att_masks = encoder_mask[:, :att_size]

            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = encoder_outputs.view(-1, att_size, encoder_outputs.size(-1))  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res


class CoAttention(nn.Module):
    def __init__(self, text_feat_size, img_feat_size, input_type='text_img'):
        """Initialize model."""
        super(CoAttention, self).__init__()
        self.text_feat_size = text_feat_size
        self.img_feat_size = img_feat_size
        self.input_type = input_type
        assert input_type in ['text_img', 'img_text', 'text_text']
        self.v_text = nn.Linear(text_feat_size, 1, bias=False)
        self.v_img = nn.Linear(img_feat_size, 1, bias=False)
        self.text2img_project = nn.Linear(text_feat_size, img_feat_size, bias=False)
        self.img2text_project = nn.Linear(img_feat_size, text_feat_size, bias=False)
        self.img_project = nn.Linear(img_feat_size, img_feat_size)
        self.text_project = nn.Linear(text_feat_size, text_feat_size)
        self.softmax = MaskedSoftmax(dim=1)
        self.linear = nn.Linear(text_feat_size + img_feat_size, text_feat_size)

    def text_att_scores(self, text_feat, img_feats):
        batch_size, img_num, img_feat_size = list(img_feats.size())
        batch_size, text_feat_size = list(text_feat.size())

        img_feats_ = img_feats.view(-1, img_feat_size)  # [batch_size*img_num, img_feat_size]
        img_feature = self.img2text_project(img_feats_)  # [batch_size*img_num, text_feat_size]

        # Project decoder state: text_feats (in our case)
        text_feature = self.text_project(text_feat)  # [batch_size, text_feat_size]
        text_feature_expanded = text_feature.unsqueeze(1).expand(batch_size, img_num, text_feat_size).contiguous()
        text_feature_expanded = text_feature_expanded.view(-1, text_feat_size)  # [batch_size*img_num, text_feat_size]

        # sum up attention features
        att_features = img_feature + text_feature_expanded  # [batch_size*img_num, text_feat_size]
        e = F.tanh(att_features)  # [batch_size*img_num, text_feat_size]
        scores = self.v_text(e)  # [batch_size*img_num, 1]
        scores = scores.view(-1, img_num)  # [batch_size, img_num]
        return scores

    def addi_att_scores(self, img_feat, text_feats):
        batch_size, max_src_len, text_feat_size = list(text_feats.size())
        batch_size, img_feat_size = list(img_feat.size())

        text_feats_ = text_feats.view(-1, text_feat_size)  # [batch_size*max_src_len, text_feat_size]
        text_feature = self.text2img_project(text_feats_)  # [batch_size*max_src_len, img_feat_size]

        # Project decoder state: text_feats (in our case)
        img_feature = self.img_project(img_feat)  # [batch_size, img_feat_size]
        img_feature_expanded = img_feature.unsqueeze(1).expand(batch_size, max_src_len, img_feat_size).contiguous()
        img_feature_expanded = img_feature_expanded.view(-1, img_feat_size)  # [batch_size*max_src_len, img_feat_size]

        # sum up attention features
        att_features = text_feature + img_feature_expanded  # [batch_size*max_src_len, img_feat_size]
        e = F.tanh(att_features)  # [batch_size*max_src_len, img_feat_size]
        scores = self.v_img(e)  # [batch_size*max_src_len, 1]
        scores = scores.view(-1, max_src_len)  # [batch_size, max_src_len]
        return scores

    def forward(self, text_feats, addi_feats, src_mask, addi_mask=None):
        # Text
        batch_size, addi_num, addi_feat_size = list(addi_feats.size())
        batch_size, max_src_len, text_feat_size = list(text_feats.size())

        if self.input_type in ['text_img', 'text_text']:
            text_feat = masked_max(text_feats, src_mask, dim=1)
        elif self.input_type in ['img_text']:
            text_feat = torch.mean(text_feats, dim=1)

        text_scores = self.text_att_scores(text_feat, addi_feats)

        text_att_dist = F.softmax(text_scores, dim=1)
        if addi_mask is not None:
            att_masks = addi_mask[:, :addi_feats.size(1)]
            text_att_dist = text_att_dist * att_masks.view(-1, addi_feats.size(1)).float()
            text_att_dist = text_att_dist / text_att_dist.sum(1, keepdim=True)  # normalize to 1

        text_att_dist = text_att_dist.unsqueeze(1)  # [batch_size, 1, img_num]
        addi_feats = addi_feats.view(-1, addi_num, addi_feat_size)  # batch_size, img_num, img_feat_size]
        addi_context = torch.bmm(text_att_dist, addi_feats)  # [batch_size, 1, img_feat_size]
        addi_context = addi_context.squeeze(1)  # [batch_size, img_feat_size]

        addi_feat = masked_max(addi_feats, addi_mask, dim=1)
        addi_scores = self.addi_att_scores(addi_feat, text_feats)

        addi_att_dist = F.softmax(addi_scores, dim=1)  # (batch, att_size)

        if src_mask is not None:
            att_masks = src_mask[:, :text_feats.size(1)]
            addi_att_dist = addi_att_dist * att_masks.view(-1, text_feats.size(1)).float()
            addi_att_dist = addi_att_dist / addi_att_dist.sum(1, keepdim=True)  # normalize to 1

        addi_att_dist = addi_att_dist.unsqueeze(1)  # [batch_size, 1, max_src_len]
        text_feats = text_feats.view(-1, max_src_len, text_feat_size)  # [batch_size, max_src_len, text_feat_size]
        text_context = torch.bmm(addi_att_dist, text_feats)  # [batch_size, 1, text_feat_size]
        text_context = text_context.squeeze(1)  # [batch_size, text_feat_size]

        combined_features = torch.cat([addi_context, text_context], dim=1)
        combined_features = self.linear(combined_features)
        return combined_features


class MyMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_kv, dropout=0.1, need_mask=False, is_regu=False):
        super(MyMultiHeadAttention, self).__init__()
        self.need_mask = need_mask
        self.is_regu = is_regu
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_kv, d_kv, dropout=dropout, is_regu=is_regu)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model, dropout=dropout)

    def forward(self, q, k, v, mask=None):
        # q: [batch_size, d_model] ==>  k: [batch_size, 1, d_model]
        # mask: [batch_size, seq_len] == > [batch_size, 1, seq_len]
        # when there is only one query, we need to expand the dimension
        if len(q.shape) == 2:
            q = q.unsqueeze(1)
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len]
        if self.need_mask:
            assert mask is not None, 'Please pass the attention mask to the multi-head'

        if self.is_regu:
            enc_output, enc_slf_attn, head_diff = self.slf_attn(q, k, v, mask)
        else:
            enc_output, enc_slf_attn = self.slf_attn(q, k, v, mask)
        enc_output = self.pos_ffn(enc_output)

        # enc_output: [batch_size, 1, d_model] ==>  k: [batch_size, d_model]
        enc_output = enc_output.squeeze(1)
        if self.is_regu:
            return enc_output, enc_slf_attn, head_diff
        return enc_output, enc_slf_attn
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
from lmdbdict import lmdbdict
from lmdbdict.methods import DUMPS_FUNC, LOADS_FUNC
import os
import numpy as np
import numpy.random as npr
import random
from functools import partial

import torch
import torch.utils.data as data

import multiprocessing
import six
from torch.utils.data import TensorDataset, DataLoader, Dataset, SequentialSampler

class load_ce_data(Dataset):
    def __init__(self, text_file, img_feature_path, comment_file, com_num_use, split=None):

        total_data = json.load(open(text_file, 'r'))

        cur_split_data = total_data[split]
        self.img_feature_path = img_feature_path

        self.ids = []
        self.texts = []
        self.labels = []
        for item in cur_split_data:
            id = item['id']
            text = item["text"]
            label = item["label"]
            self.ids.append(id)
            self.texts.append(text)

            if int(label[0])==1:
                tmp_l = 0
            elif int(label[1])==1:
                tmp_l = 1
            elif int(label[2])==1:
                tmp_l = 2
            else:
                tmp_l = 3
            self.labels.append(tmp_l)


        self.ret_text_comment_all = json.load(open(comment_file, 'r'))
        self.com_num_use = com_num_use


    def __len__(self):
        return (len(self.ids))

    def __getitem__(self, i):
        id = self.ids[i]
        text = self.texts[i]
        label = self.labels[i]
        img_feature = np.load(os.path.join(self.img_feature_path, id+'.npy'))

        ret_texts = self.ret_text_comment_all[id+'.jpg']["consensus_text"]
        ret_coms = self.ret_text_comment_all[id+'.jpg']["consensus_com"]
        ret_texts_new = []
        for ret_text in ret_texts:
            # ensure each ret_text contains at least 1 words
            if len(ret_text.split(' ')) > 0 and len(ret_texts_new) < self.com_num_use:
                ret_texts_new.append(ret_text)
        if len(ret_texts_new)==0:
            ret_texts_new = ['None']
        # ensure the last ret_texts_new contains 5 texts
        if len(ret_texts_new) < self.com_num_use:
            for i in range(self.com_num_use - len(ret_texts_new)):
                ret_texts_new.append(ret_texts_new[i])
        ret_coms_new = []
        for ret_com in ret_coms:
            # ensure each ret_text contains at least 1 words
            if len(ret_com.split(' ')) > 0 and len(ret_coms_new) < self.com_num_use:
                ret_coms_new.append(ret_com)
        # ensure the last ret_texts_new contains 5 texts
        if len(ret_coms_new) < self.com_num_use:
            for i in range(self.com_num_use - len(ret_coms_new)):
                ret_coms_new.append(ret_coms_new[i])


        sample = (img_feature, text, label, ret_texts_new, ret_coms_new)

        return sample


class load_self_data(Dataset):
    def __init__(self, text_file, self_img_feature_path, comment_file, com_num_use, self_train_num,
                 split=None):

        total_data = json.load(open(text_file, 'r'))

        cur_split_data = total_data[split]
        self.img_feature_path = self_img_feature_path

        self.ids = []
        for item in cur_split_data:
            id = item['id']
            for i in range(self_train_num):
                self.ids.append(id + '-' + str(i))

        self.ret_text_comment_all = json.load(open(comment_file, 'r'))
        self.com_num_use = com_num_use
        self.self_train_num = self_train_num

    def __len__(self):
        return (len(self.ids))

    def __getitem__(self, i):
        id = self.ids[i]  # '1023994392602968065-0'

        img_feature = np.load(os.path.join(self.img_feature_path, id + '.npy'))

        real_id, ptr_idx = id.split('-')[0], id.split('-')[1]
        ptr_idx = int(ptr_idx)
        ret_texts = self.ret_text_comment_all[real_id + '.jpg']["consensus_text"]
        ret_coms = self.ret_text_comment_all[real_id + '.jpg']["consensus_com"]

        ret_texts_new = []
        for ret_text in ret_texts:
            # ensure each ret_text contains at least 1 words
            if len(ret_text.split(' ')) > 0 and len(ret_texts_new) < self.self_train_num:
                ret_texts_new.append(ret_text)
        if len(ret_texts_new) == 0:
            ret_texts_new = ['None']
        # ensure the last ret_texts_new contains self_train_num texts
        if len(ret_texts_new) < self.self_train_num:
            for i in range(self.self_train_num - len(ret_texts_new)):
                ret_texts_new.append(ret_texts_new[i])
        ret_coms_new = []
        for ret_com in ret_coms:
            # ensure each ret_text contains at least 1 words
            if len(ret_com.split(' ')) > 0 and len(ret_coms_new) < self.com_num_use:
                ret_coms_new.append(ret_com)
        # ensure the last ret_texts_new contains 5 texts
        if len(ret_coms_new) < self.com_num_use:
            for i in range(self.com_num_use - len(ret_coms_new)):
                ret_coms_new.append(ret_coms_new[i])

        # get the text related to the img
        ret_text = ret_texts_new[ptr_idx]
        # ret_text_ocr = ret_text + ' ' + ocr_text
        # get the revise comments for the student model
        ret_coms_new_revise = []
        replace_idx = [random.random() > 0.5 for _ in
                       range(self.com_num_use)]  # the proportion of each com being replaced is 0.2
        for r_idx, com_ in zip(replace_idx, ret_coms_new):
            if r_idx:  # dropout the com
                continue
            else:
                ret_coms_new_revise.append(com_)
        # ensure that tmp_com_new has at least one text
        if len(ret_coms_new_revise) == 0:
            ret_coms_new_revise.append(com_)
        for i in range(self.com_num_use - len(ret_coms_new_revise)):
            ret_coms_new_revise.append(ret_coms_new_revise[i])

        sample = (img_feature, ret_text, ret_coms_new, ret_coms_new_revise)

        return sample

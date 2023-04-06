import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader,random_split
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import time
import os
import PIL
import pickle
import json
import random


"""
read text file, find corresponding image path
"""
def load_ce_data(img_feature_path, text_comment_file, com_num_use):
    all_data = []
    text_comment_data = json.load(open(text_comment_file, 'r'))
    del_items = ["sarcasm", "sarcastic", "reposting", "<url>", "joke", "humour", "humor", "jokes", "irony", "ironic", "exgag"]
    all_imgs = os.listdir(img_feature_path)
    for split in ['train', 'valid', 'test']:
        data_set = dict()
        file = open(os.path.join("./data/", split + ".txt"), "rb")
        for line in file:
            content = eval(line)
            image = content[0]
            sentence = content[1]
            group = content[2]

            if image + '.npy' in all_imgs:
                del_flag = False

                for del_item in del_items:
                    if del_item in sentence:
                        del_flag = True
                        break
                if not del_flag:

                    text_comment = text_comment_data[image + '.jpg']
                    ret_texts, ret_coms = text_comment["consensus_text"], text_comment["consensus_com"]

                    ret_texts_new = []
                    for ret_text in ret_texts:
                        # ensure each ret_text contains at least 1 words
                        if len(ret_text.split(' '))>0 and len(ret_texts_new)< com_num_use:
                            ret_texts_new.append(ret_text)
                    # ensure the last ret_texts_new contains 5 texts
                    if len(ret_texts_new)< com_num_use:
                        for i in range(5-len(ret_texts_new)):
                            ret_texts_new.append(ret_texts_new[i])

                    ret_coms_new = []
                    for ret_com in ret_coms:
                        # ensure each ret_text contains at least 1 words
                        if len(ret_com.split(' ')) > 0 and len(ret_coms_new) < com_num_use:
                            ret_coms_new.append(ret_com)
                    # print(image)
                    if len(ret_coms_new)==0:
                        ret_coms_new = ['None']
                    # ensure the last ret_texts_new contains 5 texts
                    if len(ret_coms_new) < com_num_use:
                        for i in range(5 - len(ret_coms_new)):
                            ret_coms_new.append(ret_coms_new[i])

                    data_set[int(image)] = {"text": sentence, "group": group, 'ret_texts':ret_texts_new, 'ret_coms':ret_coms_new}
        all_data.append(data_set)
    train_data, valid_data, test_data = all_data[0], all_data[1], all_data[2]
    return train_data, valid_data, test_data

def load_self_train_data(ce_train_list, text_comment_file, com_num_use, self_train_num):
    self_train_data = {}
    text_comment_data = json.load(open(text_comment_file, 'r'))
    del_items = ["sarcasm", "sarcastic", "reposting", "<url>", "joke", "humour", "humor", "jokes", "irony", "ironic",
                 "exgag"]
    for key, value in text_comment_data.items():

        # select self_train_num image-text pairs for self-training
        img_id = os.path.join(key.split('.')[0])
        # 只取ce-train data对应的相似数据

        if int(img_id) in ce_train_list:
            ret_texts, ret_coms = value["consensus_text"], value["consensus_com"]

            ret_texts_new = []
            for ret_text in ret_texts:
                # ensure each ret_text contains at least 1 words
                if len(ret_text.split(' ')) > 0 and len(ret_texts_new) < self_train_num:
                    ret_texts_new.append(ret_text)
            # ensure the last ret_texts_new contains 5 texts
            if len(ret_texts_new) < self_train_num:
                for i in range(self_train_num - len(ret_texts_new)):
                    ret_texts_new.append(ret_texts_new[i])

            ret_coms_new = []
            for ret_com in ret_coms:
                # ensure each ret_text contains at least 1 words
                if len(ret_com.split(' ')) > 0 and len(ret_coms_new) < com_num_use:
                    ret_coms_new.append(ret_com)
            # ensure the last ret_texts_new contains 5 texts
            if len(ret_coms_new) == 0:
                ret_coms_new = ['None']
            if len(ret_coms_new) < com_num_use:
                for i in range(5 - len(ret_coms_new)):
                    ret_coms_new.append(ret_coms_new[i])

            for i, ret_text in enumerate(ret_texts_new):
                del_flag = False
                # for del_item in del_items:   # del the sentence which contains the explicit sarcasm words
                #     if del_item in ret_text:
                #             del_flag = True
                #             break
                # if not del_flag:
                self_train_data[img_id+'-'+str(i)] = {"text": ret_text, "group": 2, 'ret_texts': ret_texts_new, 'ret_coms': ret_coms_new}

    return self_train_data

"""
load all training data 
"""
# load image labels for ce data
def load_image_labels():
    # get labels
    img2labels=dict()
    with open(os.path.join("./data/attr","img_to_five_words.txt"),"rb") as file:
        for line in file:
            content=eval(line)
            img2labels[int(content[0])]=content[1:]
    return img2labels

# load image labels for ce data
def load_image_self_train_labels():
    # get labels
    img2labels=dict()
    with open(os.path.join("./data/retrieval","sarcasm_retrieval_attributes.txt"),"r") as file:
        for line in file:
            id, attributes = line.strip().split(': ')

            img2labels[str(id.split('.')[0])]=attributes.split(';')
    return img2labels

# save to dataloader
class my_data_set(Dataset):
    def __init__(self, data, opt):
        self.data=data
        TEXT_LENGTH = opt.TEXT_LENGTH
        self.img_feature_path = opt.img_feature_path

        self.image_ids=list(data.keys())


        self.img2labels = load_image_labels()


    # load image feature data - resnet 50 result
    def __image_feature_loader(self,id):
        attribute_feature = np.load(os.path.join(self.img_feature_path,str(id)+".npy"))
        return torch.from_numpy(attribute_feature)

    # load attribute feature data - 5 words label
    def __attribute_loader(self,id):
        labels= self.img2labels[id]
        labels = ' '.join(labels)
        return labels

    def __text_loader(self,id):
        return self.data[id]["text"]
    def __ret_text_loader(self,id):
        return self.data[id]["ret_texts"]
    def __ret_com_loader(self,id):
        return self.data[id]["ret_coms"]

    def text_loader(self,id):
        return self.data[id]["text"]
    def label_loader(self,id):
        return self.img2labels[id]


    def __getitem__(self, index):
        id=self.image_ids[index]
        text = self.__text_loader(id)
        image_feature = self.__image_feature_loader(id)
        attribute = self.__attribute_loader(id)
        # ret_text_index = [self.__ret_text_index_loader(id)]
        ret_coms = self.__ret_com_loader(id)
        ret_texts = self.__ret_text_loader(id)
        group = self.data[id]["group"]
        # return text_index,image_feature,attribute_index, ret_text_index, ret_com_index, group,id
        return text,image_feature,attribute, ret_coms, ret_texts, group,id

    def __len__(self):
        return len(self.image_ids)

# save to dataloader
class self_train_my_data_set(Dataset):
    def __init__(self, data, opt):
        self.data = data
        self.img_feature_path = opt.img_feature_self_train_path
        self.image_ids=list(data.keys())
        self.img2labels = load_image_self_train_labels()
        self.com_num_use = opt.com_num_use

    # load image feature data - resnet 50 result
    def __image_feature_loader(self, id):
        attribute_feature = np.load(os.path.join(self.img_feature_path, str(id) + ".npy"))
        return torch.from_numpy(attribute_feature)

    # load attribute feature data - 5 words label
    def __attribute_loader(self, id):
        labels = self.img2labels[id]
        labels = ' '.join(labels)
        return labels

    def __text_loader(self, id):
        return self.data[id]["text"]

    def __ret_text_loader(self, id):
        return self.data[id]["ret_texts"]

    def __ret_com_loader(self, id):
        return self.data[id]["ret_coms"]

    def __ret_com_revise_loader(self, id):
        return self.data[id]["ret_com_revise"]

    def text_loader(self, id):
        return self.data[id]["text"]

    def label_loader(self, id):
        return self.img2labels[id]




    def __getitem__(self, index):
        id=self.image_ids[index]
        text = self.__text_loader(id)
        image_feature = self.__image_feature_loader(id)
        attribute = self.__attribute_loader(id)
        # ret_text_index = [self.__ret_text_index_loader(id)]
        ret_coms = self.__ret_com_loader(id)

        # get the revise comments for the student model
        ret_coms_revise = []
        replace_idx = [random.random() > 0.5 for _ in
                       range(self.com_num_use)]  # the proportion of each com being replaced is 0.2
        for r_idx, com_ in zip(replace_idx, ret_coms):
            if r_idx:  # dropout the com
                continue
            else:
                ret_coms_revise.append(com_)
        # ensure that tmp_com_new has at least one text
        if len(ret_coms_revise) == 0:
            ret_coms_revise.append(com_)
        for i in range(self.com_num_use - len(ret_coms_revise)):
            ret_coms_revise.append(ret_coms_revise[i])


        group = self.data[id]["group"]
        # return text_index,image_feature,attribute_index, ret_text_index, ret_com_index, ret_com_index_revise, group,id
        return text, image_feature, attribute, ret_coms, ret_coms_revise, group,id


    def __len__(self):
        return len(self.image_ids)









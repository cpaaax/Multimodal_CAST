import json
import os
from multiprocessing import Pool
import math
import numpy as np
from tqdm import tqdm
import re
from ekphrasis.classes.segmenter import Segmenter
import shutil




def clean_str(string,is_com = False, use=True, year = None):

    # delete the unicode characters
    string = string.encode('ascii', 'ignore').decode('ascii')

    new_string = []
    string_words = string.strip().replace('\n', ' ').split(' ')
    for w in string_words:
        if '@' in w:
            continue
        elif '#' in w:
            w = w.replace('#', '')
            new_string.append(seg_tw.segment(w))
        else:
            new_string.append(w)


    string = ' '.join(new_string)
    if not use: return string
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.replace("\\(", '').replace("\\)", '').replace("\\?", '')

    string = string.replace('.', ' . ').replace(',', ' , ').replace('!', ' ! ').replace('?', ' ? ') \
        .replace('?', ' ? ').replace('\'', ' \' ').replace(':', ' : ').replace('\\', ' \\ ').replace('/', ' / ')
    string = string.replace('\' s', '\'s').replace('\' d', '\'d').replace('\' re', '\'re').replace('\' t', '\'t') \
        .replace('\' m', '\'m')
    while '  ' in string:
        string = string.replace('  ', ' ')

    del_str = ['\"']
    for s in del_str:
        string = string.replace(s, '')
    string = string.strip()

    return string.lower()







def process_text(text_paths, comment_paths):
    # load all retrieved texts
    text_dict = {}
    comment_dict = {}

    for text_path, comment_path in tqdm(zip(text_paths, comment_paths), total=len(text_paths)):
        date = '/'.join(text_path.split('/')[-3:])
        with open(text_path+'.txt', 'r') as f:
            lines = f.readlines()

        tmp = {}
        for line in lines:
            text = line.strip()
            try:
                img_id, text = text.split(' #*#*#*#*#* ')
            except:
                img_id, text = text.split(' #*#*#*#*#*')
            tmp[img_id] = text
        text_dict[date] = tmp

        # load comment
        comment_data = json.load(open(comment_path+'.json', 'r'))
        comment_new = {}

        for id, com in comment_data.items():
            com = com["comment"]
            if len(com)!=0:
                comment_new[id] = com
        comment_dict[date] = comment_new
    return text_dict, comment_dict



def multi_process(text_paths, comment_paths, max_process):
    # num_process = min(multiprocessing.cpu_count(),max_process)
    num_process = max_process

    path_nums = len(text_paths)
    interval = int(math.ceil(path_nums / num_process))
    process_label = np.arange(0,path_nums,interval)

    p = Pool()
    result = []
    text_all = {}
    comment_all = {}
    for i,lab  in enumerate(process_label):

        if i == len(process_label)-1:
            text_info_part = text_paths[process_label[i]:]
            comment_info_part = comment_paths[process_label[i]:]
        else:
            text_info_part = text_paths[process_label[i]:process_label[i+1]]
            comment_info_part = comment_paths[process_label[i]:process_label[i+1]]
        result.append(p.apply_async(process_text, args=(text_info_part, comment_info_part)))
    p.close()
    p.join()

    for i in result:
        text_data, comment_data = i.get()
        text_all.update(text_data)
        comment_all.update(comment_data)
    return text_all, comment_all


def clear_comment(coms, year=None):
    out_coms = []
    for com in coms[:5]:
        com = com["user"]
        if len(com.split(' '))<3:
            continue
        else:
            new_com = clean_str(com, is_com=True, year = year)
        if new_com not in out_coms:
                out_coms.append(new_com)
    return out_coms


def copy_img(img_root_path,self_train_img_path,ret_data_all, top_k_self_training):
    for query_id, ret_result in tqdm(ret_data_all.items()):
        assert top_k_self_training < len(ret_result), "print not enough retrieval data for self-training"
        for i, img in enumerate(ret_result[:top_k_self_training]):
            img_id, date = list(img.items())[0]
            year, month, day = date.split('/')
            img_path = os.path.join(img_root_path, year,'original_image_all',month,day, img_id+'.jpg')
            target_path = os.path.join(self_train_img_path, query_id.split('.')[0] + '-' + str(i) + '.jpg')
            shutil.copy(img_path, target_path)



def get_orig_img_text_comemnt(img_root_path,self_train_img_path, ret_data_all, orignal_text_path, orignal_comment_path,
                              ret_text_comment_save_path, top_k_self_training, top_k_retrieval):
    # select similar top_k_self_training posts as self-training data
    copy_img(img_root_path,self_train_img_path,ret_data_all, top_k_self_training)
    print("finish copying imgs")
    # first get all original text dates of all years
    date_all = []
    for query_id, ret_result in ret_data_all.items():
        date = [ list(i.values())[0] for i in ret_result]
        date_all.extend(date)
    date_all = set(date_all)
    #then read all text files corresponding to the dates
    text_paths = [os.path.join(orignal_text_path, date) for date in date_all]
    comment_paths = [os.path.join(orignal_comment_path, date) for date in date_all]
    text_all, comment_all = process_text(text_paths, comment_paths)
    # then get related text and comments based on the retrieval data
    ret_text_comment_all = {}
    for query_id, ret_result in tqdm(ret_data_all.items()):
        cur_text_all = []
        cur_com_all = []
        already_id = []
        for cur_result in ret_result[:top_k_retrieval]:
            cur_id, cur_date = list(cur_result.keys())[0], list(cur_result.values())[0]
            if cur_id not in already_id:  # delete the same retrieved img
                cur_text = text_all[cur_date][cur_id]
                cur_text = clean_str(cur_text)
                cur_com =  comment_all[cur_date][cur_id]
                cur_com = clear_comment(cur_com)

                # if cur_text not in cur_text_all and cur_text!='':  # delete the same retrieved text
                #     cur_text_all.append(cur_text)
                if cur_text!='':  # delete the same retrieved text
                    cur_text_all.append(cur_text)
                for com in cur_com:               # delete the same retrieved text
                    if com not in cur_com_all:
                        cur_com_all.append(com)
                already_id.append(cur_id)
        ret_text_comment_all[query_id] = {'ret_text':cur_text_all, 'ret_com':cur_com_all}
    json.dump(ret_text_comment_all, open(ret_text_comment_save_path, 'w'))




if __name__=='__main__':
    # get the original text and comment based on the retrieved img ids
    ret_result_save_path = './faiss_save/ITR/retrieval_results_index.json'
    ret_data_all = json.load(open(ret_result_save_path, 'r'))

    ret_text_comment_save_path = './faiss_save/ITR/retrieval_results_text_comment.json'
    img_root_path = '/home/xcp/disk_share/raw_social_dataset'
    self_train_img_path = '../self_train_data/ITR'
    if not os.path.exists(self_train_img_path):
        os.makedirs(self_train_img_path)
    # load the tool used to process hashtags
    seg_tw = Segmenter(corpus="twitter")

    orignal_comment_path = '../comment_data'
    max_process = 8
    top_k_retrieval = 10
    top_k_self_training = 10

    orignal_text_path = '/home/xcp/disk_share/raw_social_dataset'
    get_orig_img_text_comemnt(img_root_path,self_train_img_path, ret_data_all, orignal_text_path, orignal_comment_path,
                              ret_text_comment_save_path, top_k_self_training, top_k_retrieval)

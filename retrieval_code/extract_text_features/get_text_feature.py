import torch
import json
import os
import numpy as np
from tqdm import tqdm
from random import seed,choice,sample
import torch
import torchvision.models as models
import torch.nn as nn
import torch.multiprocessing as mp
import math
from simcse import SimCSE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_exist_files(save_path, data):
    exist_data = os.listdir(save_path)
    exist_data = sorted(exist_data)
    candidate_data = [i for i in data if i not in exist_data]
    print(candidate_data)
    return candidate_data

def extract_text_feature(posts, save_day_path):
    posts_all = {}
    for post in posts:
        post_split = post.split(' #*#*#*#*#* ')
        id = post_split[0]
        text = post_split[1].strip()
        posts_all[id] = text
    ids = list(posts_all.keys())
    texts = list(posts_all.values())


    with torch.no_grad():
        text_feature = model.encode(texts, batch_size=384)
    # save the text feature with the name of post
    assert len(ids) == text_feature.size()[0]
    for i, id in enumerate(ids):
        np.save(os.path.join(save_day_path, id + '.npy'), text_feature[i].data.cpu().float().numpy())





if __name__=='__main__':
    # assume the path of downloaded original text is
    #  '/home/xcp/disk_share/raw_social_dataset/year/original_text_all/month/day.txt'
    # the format of data in txt is
    # 1I83HRiHMG.jpg #*#*#*#*#* Happy New Year 2017 !\n
    # jiRW9Kz1.jpg #*#*#*#*#* Beyoncé looking flawless at the Wynn in Las Vegas!\n
    # ......
    text_root_path = '/home/xcp/disk_share/raw_social_dataset'
    save_root_path = '../text_fc_feature'
    years = ['2014', '2015', '2016', '2017', '2018', '2018_new', '2019', '2019_new']
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

    # model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

    for year in years:
        months = sorted(os.listdir(os.path.join(text_root_path,year, 'original_text_all')))
        text_month_paths = [os.path.join(text_root_path,year,i) for i in months]
        for month in tqdm(text_month_paths):
            save_month_path = os.path.join(save_root_path, year, month.split('/')[-1])
            # if not os.path.exists(save_month_path):
            #     os.makedirs(save_month_path)

            days = os.listdir(month)
            # days = detect_exist_files(save_month_path, days)
            day_files = sorted([os.path.join(month, day) for day in days])

            for day_file in tqdm(day_files):
                day = day_file.split('/')[-1].split('.')[0]
                save_day_path = os.path.join(save_month_path,day)
                if not os.path.exists(save_day_path):
                    os.makedirs(save_day_path)
                # read original text file, format: 1I83HRiHMG.jpg #*#*#*#*#* Happy New Year 2017 !
                with open(day_file, 'r') as f:
                    posts = f.readlines()
                if len(posts) == 0:
                    continue
                extract_text_feature(posts, save_day_path)
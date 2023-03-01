import json
import os
import h5py
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
import math


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



def process(img_year_path, comment_year_path, img_days, comment_days, month):
    img_feat_all = []
    img_order_all = []
    for img_day, comment_day in zip(img_days, comment_days):
        img_day_path = os.path.join(img_year_path, month, img_day)
        comment_day_path = os.path.join(comment_year_path, month, comment_day)
        date = os.path.join(comment_year_path.split('/')[-1], month, comment_day.split('.')[0])
        with open(comment_day_path, 'r') as f:
            comments = json.load(f)
            for  com_key, com_value in comments.items():
                # if img has related comments, load the image features and store the image name
                if com_value["comment"] !=[]:
                    if os.path.exists(os.path.join(img_day_path, com_key+'.npy')):
                        img_feat = np.load(os.path.join(img_day_path, com_key+'.npy'), allow_pickle=True)
                        img_feat_all.append(img_feat)
                        img_order_all.append({com_key:date})
    return img_feat_all, img_order_all





def combine_data(data_path, save_path, comment_path, max_process):
    months = sorted(os.listdir(data_path))
    data_all = []
    order_all = []

    order_all_last = {}
    year = data_path.split('/')[-1]
    num_process = min(multiprocessing.cpu_count(), max_process)
    for month in tqdm(months):
        days = sorted(os.listdir(os.path.join(data_path, month)))
        comment_files = sorted(os.listdir(os.path.join(comment_path, month)))
        # for the condition of not crawling the comments for the day
        if len(comment_files)<len(days):
            days = days[:len(comment_files)]
        for i, comment_file in enumerate(comment_files):
            day = days[i]
            assert day==comment_file.split('.')[0]

        days_num = len(days)
        interval = int(math.ceil(days_num / num_process))
        img_info_parts = list(chunks(days,interval))
        comment_info_parts = list(chunks(comment_files, interval))

        p = Pool()
        month_all = []
        for img_info_part,comment_info_part in zip(img_info_parts, comment_info_parts):
            month_all.append(p.apply_async(process, args = (data_path, comment_path, img_info_part,
                                                           comment_info_part, month)))
        p.close()
        p.join()
        for i in month_all:
            img_feats, img_orders = i.get()
            data_all.extend(img_feats)
            order_all.extend(img_orders)

    data_all = np.stack(data_all, axis=0)
    print(data_all.shape)

    for i, j in enumerate(order_all):
        order_all_last[i] = j
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with h5py.File(os.path.join(save_path, year+'_feature_all.h5'), 'w') as f:
        f.create_dataset(year, data=data_all)
    with open(os.path.join(save_path, year+'_order_all.json'), 'w') as f:
        json.dump(order_all_last, f)

if __name__=="__main__":
    # Assume the path of comment is '../comment_data/year/month/day.json'
    # for the data in json, the format of comments is {tweet_id:{"comment": []}}


    years = ['2014', '2015', '2016', '2017', '2018', '2018_new', '2019', '2019_new']
    data_root = '../image_fc_feature'
    save_root = '../image_fc_feature_combine'
    comment_root = '../comment_data'
    max_process = 30
    for year in years:
        data_path = os.path.join(data_root, year)
        save_path = os.path.join(save_root, year)
        comment_path = os.path.join(comment_root, year)
        combine_data(data_path, save_path, comment_path, max_process)
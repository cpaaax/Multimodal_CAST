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



def process(text_year_path, img_id_part, img_date_part):
    # load the text feats based on the image order
    text_feat_all = []
    text_order_all = []
    # img_id: fdsafdsa  img_date:2014/02/09
    for img_id, img_date in tqdm(zip(img_id_part, img_date_part), total=len(img_id_part)):
        # assert text_year_path.split('/')[-1] == img_date.split('/')[0]
        month, day = img_date.split('/')[1], img_date.split('/')[2]
        text_feat_path = os.path.join(text_year_path, month, day, img_id+'.npy')
        text_feat = np.load(text_feat_path, allow_pickle=True)
        text_feat_all.append(text_feat)
        text_order_all.append({img_id:img_date})
    return text_feat_all, text_order_all



def combine_data(data_path, save_path, max_process, img_all_order):
    num_process = min(multiprocessing.cpu_count(), max_process)
    # the image_all_order contains the img order of combined file


    # img_order_ids = list(img_all_order.keys())
    # img_order_dates = list(img_all_order.values())
    img_all_order = list(img_all_order.values())
    img_order_ids, img_order_dates = [], []
    for img_order in img_all_order:
        img_id, img_date = list(img_order.items())[0]
        img_order_ids.append(img_id)
        img_order_dates.append(img_date)

    img_num = len(img_order_ids)
    interval = int(math.ceil(img_num / num_process))
    img_id_parts = list(chunks(img_order_ids, interval))
    img_date_parts = list(chunks(img_order_dates, interval))
    p = Pool()
    data_all = []
    text_feat_all = []
    text_order_all = []
    text_order_last = {}
    for img_id_part, img_date_part in zip(img_id_parts, img_date_parts):
        data_all.append(p.apply_async(process, args=(data_path, img_id_part,
                                                      img_date_part)))

    p.close()
    p.join()
    for i in data_all:
        text_feats, text_orders = i.get()
        text_feat_all.extend(text_feats)
        text_order_all.extend(text_orders)

    text_feat_all = np.stack(text_feat_all, axis=0)
    print(text_feat_all.shape)
    for i, j in enumerate(text_order_all):
        text_order_last[i] = j
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with h5py.File(os.path.join(save_path, year+'_feature_all.h5'), 'w') as f:
        f.create_dataset(year, data=text_feat_all)
    with open(os.path.join(save_path, year+'_order_all.json'), 'w') as f:
        json.dump(text_order_last, f)


    check_data_after_combine(text_order_all, img_all_order)


def check_data_before_combine(text_data_path, img_data_path): # check whether the num of text and image are equal
    months = sorted(os.listdir(text_data_path))
    text_count = 0
    img_count = 0
    for month in tqdm(months):
        text_days = sorted(os.listdir(os.path.join(text_data_path, month)))
        img_days = sorted(os.listdir(os.path.join(img_data_path, month)))
        for text_day, image_day in zip(text_days, img_days):
            print(text_day, ' ' + str(image_day))
            assert text_day==image_day
            text_data = sorted(os.listdir(os.path.join(text_data_path, month, text_day)))
            image_data = sorted(os.listdir(os.path.join(text_data_path, month, text_day)))
            assert text_data==image_data
            text_count+=len(text_data)
            img_count+=len(image_data)
    print('img num {}, text num {}'.format(img_count, text_count))

def check_data_after_combine(text_all_order, img_all_order):

    assert text_all_order == img_all_order

if __name__=="__main__":
    # the order of text combine file should be same with image combine file. Here we use the image combine file as the base
    # to combine text file
    years = ['2014', '2015', '2016', '2017', '2018', '2018_new', '2019', '2019_new']

    image_data_root = '../image_fc_feature'
    text_data_root = '../text_fc_feature'

    image_combine_path = '../image_fc_feature_combine'
    save_root = '../text_fc_feature_combine'


    max_process = 24
    for year in years:
        text_data_path = os.path.join(text_data_root, year)
        save_path = os.path.join(save_root, year)
        img_combine_file = os.path.join(image_combine_path,year, year+'_order_all.json')
        with open(img_combine_file, 'r') as f:
            img_all_order = json.load(f)

        img_data_path = os.path.join(image_data_root, year)
        check_data_before_combine(text_data_path, img_data_path)
        combine_data(text_data_path, save_path, max_process, img_all_order)

        with open(os.path.join(save_path, year + '_order_all.json'), 'r') as f:
            text_combine_order = list(json.load(f).values())
        check_data_after_combine(text_combine_order, img_all_order)
import h5py
import torch
import json
import os
import numpy as np
from tqdm import tqdm
from imageio import imread
from random import seed,choice,sample
import torch
import torchvision.models as models
import torch.nn as nn
import torch.multiprocessing as mp
import math
from torchvision import transforms as trn

from misc.resnet_utils import myResnet
import misc.resnet as resnet


preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def Encoder(my_resnet, img_path):


    I = imread(img_path)
    # handle grayscale input images
    if len(I.shape) == 2:
        I = I[:, :, np.newaxis]
        I = np.concatenate((I, I, I), axis=2)

    I = I.astype('float32') / 255.0
    I = torch.from_numpy(I.transpose([2, 0, 1])).cuda()
    I = preprocess(I)
    with torch.no_grad():
        tmp_fc, tmp_att = my_resnet(I, att_size=14)
    return tmp_fc, tmp_att


def select_comment_image(day_file, save_root_path):
    with open(day_file, 'r') as f:
        data = json.load(f)
    # raw_out_images = {}
    # for img_id, value in data.items():
    #     if len(value)>0:
    #         raw_out_images[img_id]= value
    # return raw_out_images
    return data



def input_data_file(img_files,save_root_path):
    net = getattr(resnet, 'resnet101')()
    net.load_state_dict(torch.load(os.path.join('imagenet_weights', 'resnet101_caffe.pth')))
    my_resnet = myResnet(net)
    my_resnet.cuda()
    my_resnet.eval()
    del_img = []
    for img in tqdm(img_files):
        img_name = img.split('/')[-1].split('.')[0]
        day = img.split('/')[-2]
        month = img.split('/')[-3]
        year = img.split('/')[-5]
        save_img_path = os.path.join(save_root_path, year, month, day)
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)

        try:
            fc, att = Encoder(my_resnet, img)
            np.save(os.path.join(save_img_path, img_name + '.npy'), fc.data.cpu().float().numpy())
        except:
            print('image {} not exist'.format(img_name))
            del_img.append(img_name)
            continue
    print(del_img)

def detect_exist_files(save_path, data):
    exist_data = os.listdir(save_path)
    exist_data = sorted(exist_data)
    candidate_data = [i for i in data if i not in exist_data]
    print(candidate_data)
    return candidate_data






def multi_process(day_path, image_root_path, save_root_path,max_process):
    # num_process = min(multiprocessing.cpu_count(),max_process)
    imgs = os.listdir(day_path)
    imgs = [os.path.join(day_path, img) for img in imgs]
    num_process = max_process
    img_nums = len(imgs)
    interval = int(math.ceil(img_nums / num_process))
    process_label = np.arange(0,img_nums,interval)

    # assert  len(process_label)==num_process,print(img_month_path,day_nums,interval, process_label, num_process, 'process_label not equal num_process')  # equal
    # p = Pool()
    restlt_ = []
    result_last = []
    pool_list = []


    for i,lab  in enumerate(process_label):

        if i == len(process_label)-1:
            img_info_part = imgs[process_label[i]:]
        else:
            img_info_part = imgs[process_label[i]:process_label[i+1]]

        p = mp.Process(target=input_data_file, args=(img_info_part, save_root_path))
        p.start()
        pool_list.append(p)
    for p in pool_list:
        p.join()

if __name__=='__main__':
    torch.multiprocessing.set_start_method("spawn")
    # assume the path of downloaded original images is
    #  '/home/xcp/disk_share/raw_social_dataset/year/original_image_all/month/day/*.jpg' where * indicates the tweet id
    image_root_path = '/home/xcp/disk_share/raw_social_dataset'
    save_root_path = '../image_fc_feature'
    years = ['2014', '2015', '2016', '2017', '2018', '2018_new', '2019', '2019_new']

    max_process = 8
    for year in years:
        img_cur_path = os.path.join(image_root_path, year, 'original_image_all')
        month_paths = sorted(os.listdir(img_cur_path))
        month_paths = [os.path.join(img_cur_path,i) for i in month_paths]
        for month in tqdm(month_paths):
            days = os.listdir(month)
            save_path = os.path.join(save_root_path, year, month.split('/')[-1])
            if os.path.exists(save_path):
                days = detect_exist_files(save_path, days)

            day_paths = sorted([os.path.join(month, day) for day in days])
            for day_path in tqdm(day_paths):
                img_nums = os.listdir(day_path)
                if len(img_nums)==0:
                    print('there are no image in {}'.format(day_path))
                    continue
                multi_process(day_path, image_root_path, save_root_path,max_process)

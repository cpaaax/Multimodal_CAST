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
    net = getattr(resnet, 'resnet152')()
    net.load_state_dict(torch.load(os.path.join('imagenet_weights', 'resnet152.pth')))
    my_resnet = myResnet(net)
    my_resnet.cuda()
    my_resnet.eval()
    del_img = []
    for img in tqdm(img_files):
        img_name = img.split('/')[-1].split('.')[0]
        save_img_path = save_root_path
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)

        try:
            fc, att = Encoder(my_resnet, img)
            np.save(os.path.join(save_img_path, img_name + '.npy'), att.data.cpu().float().numpy())
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






def multi_process(image_root_path, save_root_path,max_process):
    # num_process = min(multiprocessing.cpu_count(),max_process)
    imgs = os.listdir(image_root_path)
    imgs = [os.path.join(image_root_path, img) for img in imgs]
    num_process = max_process
    img_nums = len(imgs)
    interval = int(math.ceil(img_nums / num_process))
    process_label = np.arange(0,img_nums,interval)

    # assert  len(process_label)==num_process,print(img_month_path,day_nums,interval, process_label, num_process, 'process_label not equal num_process')  # equal
    # p = Pool()
    restlt_ = []
    result_last = []
    pool_list = []


    # 例如5000张img, 6个cpu, process_label为[0 834 1668 2502 3336 4170]  则最后一个cpu需要计算indxes[4170:]
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

    # image_root_path = '/media/pc/Disk_8T/EMNLP_code/sarcasm/self_train_data/images'
    # save_root_path = '/media/pc/Disk_8T/EMNLP_code/sarcasm/self_train_data/extract_features_resnet152'
    image_root_path = '/media/pc/Disk_8T/EMNLP_code/self_training_imgs_all/sarcasm'
    save_root_path = '/media/pc/Disk_8T/EMNLP_code/sarcasm/self_train_data/extract_features_resnet152_10'
    # image_root_path = '/media/pc/Disk_8T/EMNLP_code/sarcasm/multimodal-sarcasm-detection/dataset_img/dataset_image'
    # save_root_path = '/media/pc/Disk_8T/EMNLP_code/sarcasm/multimodal-sarcasm-detection/dataset_img/extract_features_resnet152'


    max_process = 4
    multi_process(image_root_path, save_root_path, max_process)

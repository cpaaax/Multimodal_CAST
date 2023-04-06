import argparse
import os
import logging
from train import train
from self_train_pf import self_train_pf, reshape_text, calculate_score
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import random
from LoadData import load_ce_data, load_self_train_data, my_data_set, self_train_my_data_set
from torch.utils.data import Dataset, DataLoader, random_split
from models.main_model import Multimodel
from tqdm import  tqdm
import opts
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def eval(opt):
    device = opt.device
    model_s = Multimodel(device, opt).to(device)
    model_s.load_state_dict(torch.load(opt.trained_model_path))
    model_s.eval()

    _, _, test_data = load_ce_data(opt.img_feature_path, opt.text_comment_file, opt.com_num_use)
    test_set = my_data_set(test_data, opt)
    test_loader = DataLoader(test_set, batch_size=opt.ce_batch_size, shuffle=False, num_workers=4, )

    with torch.no_grad():
        predict_all = []
        label_all = []
        for val_text, val_image_feature, val_attribute, val_ret_coms, val_ret_texts, val_group, val_id in tqdm(
                test_loader):
            val_ret_coms_new = reshape_text(val_ret_coms, opt.com_num_use)

            val_group = val_group.view(-1).to(torch.int64).to(device)
            val_pred = model_s(val_text, val_image_feature.to(device),
                               val_attribute, val_ret_coms_new)

            # calculate the score
            predict_all.extend(val_pred.argmax(dim=1).detach().cpu().numpy())
            label_all.extend(val_group.detach().cpu().numpy())
    predict_all = np.vstack(predict_all)
    label_all = np.stack(label_all)
    precision, recall, fscore, acc = calculate_score(predict_all, label_all)

    print("test results: test_f1=%.4f test_pre=%.4f test_rec=%.4f test_acc=%.4f" % (
        fscore, precision, recall, acc))









if __name__=="__main__":
    opt = opts.parse_opt()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modes = ['ce', 'self']

    for mode in modes:
        print('mode: {}'.format(mode))
        if mode == 'ce':
            opt.trained_model_path = './save/ce_save/model_best.pth'
        if mode == 'self':
            opt.trained_model_path = './save/selftrain_save/model_best_0.pth'
        eval(opt)








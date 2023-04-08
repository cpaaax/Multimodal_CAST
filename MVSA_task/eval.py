import os
from typing import List

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import opts
import torch
import utils
import logging
from train import train
from self_train import self_train
from models.model import MultimodalEncoder
from dataloader import  load_ce_data, load_self_data
import utils
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from optimization import BertAdam, warmup_linear
from tqdm import tqdm, trange
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import numpy as np
import opts

def eval(opt):

    test_dataset = load_ce_data(opt.text_file_path, opt.img_feature_path, opt.comment_file, opt.com_num_use,
                                split='test')
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=opt.ce_batch_size)

    model_s = MultimodalEncoder(opt).to(opt.device)
    model_s.load_state_dict(torch.load(opt.trained_model_path))
    model_s.eval()
    predictions, true_labels = [], []
    for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
        img_features, texts, labels, ret_texts, ret_coms = batch
        ret_texts_new = utils.reshape_text(ret_texts, opt.com_num_use)
        ret_coms_new = utils.reshape_text(ret_coms, opt.com_num_use)

        img_features = img_features.to(opt.device)
        labels = labels.to(opt.device)
        with torch.no_grad():
            predict_out = model_s(img_features, texts, ret_coms_new)

        predictions.append(predict_out.detach().cpu().numpy())
        true_labels.extend(labels.to('cpu').numpy())

    predict_all = np.vstack(predictions)
    label_all = np.stack(true_labels)
    f_score, acc, w_f1 = utils.calculate_score_f1(predict_all, label_all)
    test_f1 = w_f1
    print()
    print("label_0 F1  | label_1 F1  | label_2 F1  | Test f1  | Test acc")
    print(
        f"{f_score[0]:.4f}   |  {f_score[1]:.4f}  |  {f_score[2]:.4f}  |   {test_f1:.4f}  |  {acc:.4f}")


if __name__ == "__main__":

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


import os
from typing import List

import opts
import torch
import utils
import logging
from train import train
from self_train import self_train
from dataloader import  load_ce_data, load_self_data
from models.model import MultimodalEncoder
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support as score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def calculate_score(predict, target):
    # predict: [batch, class_size], target:[batch,1]
    predict_label = predict.argmax(axis=1)
    target = target.squeeze()
    precision, recall, fscore, support = score(target, predict_label, average='weighted')
    return precision, recall, fscore, support

def calculate_score_all(predict_all, label_all):
    precision, recall, fscore, support = calculate_score(predict_all, label_all)
    print('pre {:.4f}, '.format(precision), 'rec {:.4f}'.format(recall), 'f1 {:.4f}'.format(fscore))
    return precision, recall, fscore

def eval(opt):
    test_dataset = load_ce_data(opt.text_file_path, opt.img_feature_path, opt.comment_file, opt.com_num_use,
                                split='test')
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=opt.ce_train_batch_size, num_workers=4)

    model_s = MultimodalEncoder(opt).to(opt.device)
    model_s.eval()

    model_save_path = os.path.join(opt.ce_save_path_root, 'model_best.pth')
    model_s.load_state_dict(torch.load(model_save_path))

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
    precision, recall, f_score = calculate_score_all(predict_all, label_all)


if __name__ == "__main__":
    opt = opts.parse_opt()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval(opt)






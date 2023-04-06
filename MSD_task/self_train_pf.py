import json

import torch

from LoadData import load_ce_data, load_self_train_data, my_data_set, self_train_my_data_set
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import os
from models.main_model import Multimodel, SelfTrainLoss

from optimization import BertAdam
import itertools


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def getScore(p, y):
    tp = fp = tn = fn = 0
    for i in range(p.shape[0]):
        if y[i] == 1:
            if p[i] ==1:
                tp += 1
            else:

                fn += 1
        else:
            if p[i] ==1:
                fp += 1
            else:
                tn += 1
    return tp, fp, tn, fn

def getF1(p, y):
    tp, fp, tn, fn = getScore(p, y)
    try:
        pre = float(tp) / (tp+fp)
        rec = float(tp) / (tp+fn)
        f1 = 2*pre*rec / (pre+rec)
    except:
        pre = rec = f1 = 0
    acc = float(tp + tn) / (tp + fp + tn + fn)
    return pre, rec, f1, acc


def calculate_score(predict, target):
    # predict: [batch], target:[batch]
    predict_label = predict.squeeze()
    target = target.squeeze()
    pre, rec, f1, acc = getF1(predict_label, target)
    return pre, rec, f1, acc


def reshape_text(texts, num):
    # texts: [num, batch] -> [batch, num] -> [batch*num]
    batch = len(texts[0])
    texts_new = []
    for i in range(batch):
        tmp = []
        for j in range(num):
            tmp.append(texts[j][i])
        texts_new.extend(tmp)
    return texts_new


def self_train_pf(opt, logging):
    logging.info("***********************************************************************************************")
    logging.info("run experiments with seed {} self_train_num {}".format(opt.seed, opt.self_train_num))

    loss_fn = torch.nn.CrossEntropyLoss()
    ce_batch_size = opt.ce_batch_size
    self_batch_size = opt.self_batch_size
    data_shuffle = opt.data_shuffle

    # load ce_data
    ce_train_data, valid_data, test_data = load_ce_data(opt.img_feature_path, opt.text_comment_file, opt.com_num_use)
    # train, val, test, split
    ce_train_set = my_data_set(ce_train_data, opt)
    # load self_train data
    ce_train_list = list(ce_train_data.keys())
    self_train_data = load_self_train_data(ce_train_list, opt.text_comment_file, opt.com_num_use, opt.self_train_num)
    self_train_set = self_train_my_data_set(self_train_data, opt)

    # set the valid dataset
    val_set = my_data_set(valid_data, opt)
    # val_loader = DataLoader(val_set, batch_size=ce_batch_size,num_workers=4, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=ce_batch_size, shuffle=False, num_workers=4,)
    ce_train_loader = DataLoader(ce_train_set, batch_size=ce_batch_size, shuffle=True, num_workers=4,)
    selftrain_loader = DataLoader(self_train_set, batch_size=self_batch_size, shuffle=True, num_workers=4,)

    assert len(selftrain_loader)==len(ce_train_loader)
    # initialize the model
    device = opt.device
    model_t = Multimodel(device, opt).to(device)
    model_t.load_state_dict(torch.load(opt.teacher_restore_path))
    model_s = Multimodel(device, opt).to(device)
    # optimizer
    param_optimizer = list(model_s.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_self_train_optimization_steps = int(
        len(self_train_data) / opt.self_batch_size) * opt.max_epochs

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=opt.learning_rate,
                         warmup=opt.warmup_proportion,
                         t_total=num_self_train_optimization_steps)



    number_of_epoch = opt.max_epochs
    iteration = 0
    self_loss_fn = SelfTrainLoss()
    best_val_score = 0
    best_val_score = 0
    early_epoch = 0
    min_val_loss = 0
    for epoch in range(number_of_epoch):

        train_loss = 0
        correct_train = 0
        model_s.train()
        model_t.eval()

        for step, (self_batch, ce_batch) in tqdm(enumerate(zip(selftrain_loader, ce_train_loader)),
                                                 total=len(selftrain_loader)):
            text, image_feature, attribute, ret_com, ret_com_revise, group, id = self_batch

            ret_coms_new = reshape_text(ret_com, opt.com_num_use)
            ret_coms_new_revise = reshape_text(ret_com_revise, opt.com_num_use)

            with torch.no_grad():
                pred_t = model_t(text, image_feature.to(device), attribute,
                                 ret_coms_new)

            pred_s = model_s(text, image_feature.to(device), attribute,
                             ret_coms_new_revise)
            self_loss = self_loss_fn(pred_s, pred_t)

            text, image_feature, attribute, ret_com, ret_text, group, id = ce_batch
            group = group.view(-1).to(torch.int64).to(device)
            ret_coms_new = reshape_text(ret_com, opt.com_num_use)

            pred_s = model_s(text, image_feature.to(device), attribute,
                             ret_coms_new)
            ce_loss = loss_fn(pred_s, group)
            total_loss = ce_loss + opt.loss_w * self_loss

            train_loss += total_loss.item()
            correct_train += (pred_s.argmax(dim=1) == group).sum().item()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            iteration += 1



        # calculate valid loss
        valid_loss = 0
        correct_valid = 0
        model_s.eval()

        with torch.no_grad():
            predict_all = []
            label_all = []

            for val_text, val_image_feature, val_attribute, val_ret_coms, val_ret_texts, val_group, val_id in tqdm(
                    val_loader):
                val_ret_coms_new = reshape_text(val_ret_coms, opt.com_num_use)
                val_ret_texts_new = reshape_text(val_ret_texts, opt.com_num_use)

                val_group = val_group.view(-1).to(torch.int64).to(device)
                val_pred = model_s(val_text, val_image_feature.to(device),
                                 val_attribute, val_ret_coms_new)
                val_loss = loss_fn(val_pred, val_group)
                valid_loss += val_loss
                correct_valid += (val_pred.argmax(dim=1) == val_group).sum().item()

                # calculate the score
                predict_all.extend(val_pred.argmax(dim=1).detach().cpu().numpy())
                label_all.extend(val_group.detach().cpu().numpy())
        predict_all = np.vstack(predict_all)
        label_all = np.stack(label_all)
        precision, recall, fscore, acc = calculate_score(predict_all, label_all)

        cur_val_loss = valid_loss / len(val_loader)

        print("epoch/iteration: %d/%d train_loss=%.5f train_acc=%.4f valid_loss=%.5f valid_f1=%.4f valid_pre=%.4f valid_rec=%.4f valid_acc=%.4f" % (
            epoch, iteration, train_loss / len(ce_train_loader), correct_train / len(ce_train_loader) / ce_batch_size,
            valid_loss / len(val_loader),fscore, precision, recall, acc))
        logging.info(
            "epoch/iteration: %d/%d train_loss=%.5f train_acc=%.4f valid_loss=%.5f valid_f1=%.4f valid_pre=%.4f valid_rec=%.4f valid_acc=%.4f" % (
                epoch, iteration, train_loss / len(ce_train_loader), correct_train / len(ce_train_loader) / ce_batch_size,
                valid_loss / len(val_loader), fscore, precision, recall, acc))

        f_val_score =  fscore

        if not os.path.exists(opt.self_save_path):
            os.makedirs(opt.self_save_path)

        # save the model each epoch
        # torch.save(model_s.state_dict(), os.path.join(opt.self_save_path, 'model.pth'))
        # save the best model
        if f_val_score > best_val_score:
            torch.save(model_s.state_dict(), os.path.join(opt.self_save_path, 'model_best_{}.pth'.format(opt.ts_iteration)))
            best_val_score = f_val_score
            logging.info("**********best val************")


        if epoch==0:
            min_val_loss = cur_val_loss
        else:
            if cur_val_loss > min_val_loss:
                early_epoch+=1
                if early_epoch>opt.early_stopping_tolerance:
                    break
            else:
                early_epoch = 0
                min_val_loss = cur_val_loss

    iteration += 1

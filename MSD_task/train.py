import json

import torch

from LoadData import load_ce_data, my_data_set
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from tqdm import tqdm
import argparse
import os
from models.main_model import Multimodel

# device = torch.device("cpu")
import sklearn.metrics as metrics
from optimization import BertAdam

import shutil
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




def calculate_score(predict, target):
    # predict: [batch], target:[batch]
    predict_label = predict.squeeze()
    target = target.squeeze()
    pre, rec, f1, acc = getF1(predict_label, target)
    return pre, rec, f1, acc




def train(opt, logging):
    logging.info("***********************************************************************************************")
    logging.info("run experiments with seed {}".format(opt.seed))

    loss_fn = torch.nn.CrossEntropyLoss()
    batch_size = opt.ce_batch_size
    data_shuffle = opt.data_shuffle

    train_data, valid_data, test_data = load_ce_data(opt.img_feature_path, opt.text_comment_file, opt.com_num_use)
    # train, val split

    train_set = my_data_set(train_data, opt)
    val_set = my_data_set(valid_data, opt)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)



    # initilize the model
    device = opt.device
    model = Multimodel(device, opt).to(device)
    # optimizer

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(
        len(train_data) / opt.ce_batch_size) * opt.max_epochs

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=opt.learning_rate,
                         warmup=opt.warmup_proportion,
                         t_total=num_train_optimization_steps)





    number_of_epoch = opt.max_epochs
    update_modal_lr_flag = True
    iteration = 0
    best_val_score = 0
    early_epoch=0
    min_val_loss = 0

    val_epoch_score, test_epoch_score = [], []
    for epoch in range(number_of_epoch):

        train_loss = 0
        correct_train = 0
        model.train()

        for text, image_feature, attribute, ret_coms, ret_texts, group, id in tqdm(train_loader):
            # ret_text_index = torch.tensor(np.array([item.detach().cpu().numpy() for item in ret_text_index[0]],
            #                                        dtype=np.int64)).permute(1,0,2)  # [batch, 5, 30]

            ret_coms_new = reshape_text(ret_coms, opt.com_num_use)
            ret_texts_new = reshape_text(ret_texts, opt.com_num_use)
            # print(ret_com_index.detach().cpu().numpy())
            model.train()
            group = group.view(-1).to(torch.int64).to(device)
            pred = model(text, image_feature.to(device), attribute, ret_coms_new)
            loss = loss_fn(pred, group)
            train_loss += loss.item()
            correct_train += (pred.argmax(dim=1) == group).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration+=1
            # if (iteration%30)==0:
            #     logging.info("current total loss is {:.4f}".format(loss.item()))

        # calculate valid loss
        valid_loss = 0
        correct_valid = 0
        model.eval()

        with torch.no_grad():
            predict_all = []
            label_all = []
            for val_text, val_image_feature, val_attribute, val_ret_coms, val_ret_texts, val_group, val_id in tqdm(val_loader):

                val_ret_coms_new = reshape_text(val_ret_coms, opt.com_num_use)
                val_ret_texts_new = reshape_text(val_ret_texts, opt.com_num_use)

                val_group = val_group.view(-1).to(torch.int64).to(device)
                val_pred = model(val_text, val_image_feature.to(device),
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
            epoch, iteration, train_loss / len(train_loader), correct_train / len(train_loader) / batch_size,
            valid_loss / len(val_loader),fscore, precision, recall, acc))
        logging.info(
            "epoch/iteration: %d/%d train_loss=%.5f train_acc=%.4f valid_loss=%.5f valid_f1=%.4f valid_pre=%.4f valid_rec=%.4f valid_acc=%.4f" % (
                epoch, iteration, train_loss / len(train_loader), correct_train / len(train_loader) / batch_size,
                valid_loss / len(val_loader), fscore, precision, recall, acc))
        f_val_score =  fscore
        val_epoch_score.append(f_val_score)
        if not os.path.exists(opt.ce_save_path):
            os.makedirs(opt.ce_save_path)




        # save the model each epoch
        # torch.save(model.state_dict(), os.path.join(opt.ce_save_path, 'model.pth'))
        # save the best model
        if f_val_score > best_val_score:
            torch.save(model.state_dict(), os.path.join(opt.ce_save_path, 'model_best.pth'))
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



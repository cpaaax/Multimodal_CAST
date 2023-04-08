from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


import six
from six.moves import cPickle
from torch.autograd import Variable
import logging
import os
from sklearn.metrics import precision_recall_fscore_support as score
import random
import torch.nn.functional as F

from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

def pickle_load(f):
    """ Load a pickle.
    Parameters
    ----------
    f: file-like object
    """
    if six.PY3:
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)


def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: pickled object
    f: file-like object
    """
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is None:
                continue
            else:
                param.grad.data.clamp_(-grad_clip, grad_clip)

# class ClassCriterion(nn.Module):
#     def __init__(self):
#         super(ClassCriterion, self).__init__()
#         self.criterion = nn.CrossEntropyLoss()
#     def forward(self, classifier_output, trg_class, class_num):
#         classifier_loss = self.criterion(classifier_output, trg_class.squeeze())
#
#         return classifier_loss

class ClassCriterion(nn.Module):
    def __init__(self):
        super(ClassCriterion, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, classifier_output, trg_class):
        batch = trg_class.size(0)
        classifier_loss = self.criterion(classifier_output, trg_class.squeeze())
        # classifier_loss = torch.mean(classifier_loss * weight)
        return classifier_loss


class SelfTrainLoss(nn.Module):
    def __init__(self):
        super(SelfTrainLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction= 'batchmean', log_target =False)
    def forward(self, input, target, mask=None):
        # input:[batch, len, label_num], target:[batch, len, label_num] mask:[batch, len]
        input = F.log_softmax(input, dim=1)
        target = F.softmax(target, dim=1)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            input_with_mask = input*mask
            targe_with_mask = target*mask
            loss = self.criterion(input_with_mask, targe_with_mask)
        else:
            loss = self.criterion(input, target)
        return loss


def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))

def calculate_score(predict, target):
    # predict: [batch, class_size], target:[batch,1]
    predict_label = predict.argmax(axis=1)
    target = target.squeeze()
    acc = accuracy_score(target, predict_label)

    precision, recall, fscore, support = score(target, predict_label)
    return precision, recall, fscore, acc, support


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def metrics(labels, preds, argmax_needed: bool = False):
    """
    Returns the Matthew's correlation coefficient, accuracy rate, true positive rate, true negative rate, false positive rate, false negative rate, precission, recall, and f1 score

    labels: list of correct labels

    pred: list of model predictions

    argmax_needed (boolean): converts logits to predictions. Defaulted to false.
    """

    if argmax_needed == True:
        preds = np.argmax(preds, axis=1).flatten()

    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    f1 = f1_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")

    results = {
        "mcc": mcc,
        "acc": acc,
        # "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # return results, labels, preds
    return results




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


def restrict_label(predict_t):
    # predict_t:[batch, tag_num]
    batch = predict_t.size(0)
    pre_label = torch.argmax(predict_t, dim=1)
    mask = torch.zeros(batch, device=pre_label.device)
    cnt = 0
    for ptr, l in enumerate(pre_label):
        l = l.item()
        if l == 0:
           cnt+=1
           if cnt < int(0.5*batch):
               mask[ptr] = 1
        else:
            mask[ptr] = 1
    return mask

def calculate_score_f1(predict_all, label_all):
    precision, recall, fscore, acc, support = calculate_score(predict_all, label_all)
    label_all_ = np.squeeze(label_all)
    label_0 = sum(np.equal(label_all_, 0))
    label_1 = sum(np.equal(label_all_, 1))
    label_2 = sum(np.equal(label_all_, 2))
    label_sum = label_0 + label_1 + label_2
    score_weight = np.array(
        [label_0 / label_sum, label_1 / label_sum, label_2 / label_sum])
    overall_fscore = score_weight[0] * fscore[0] + score_weight[1] * fscore[1] + score_weight[2] * fscore[2]


    return fscore, acc, overall_fscore



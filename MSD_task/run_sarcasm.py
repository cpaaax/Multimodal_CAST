import argparse
import os
import logging
import opts


from train import train
from self_train_pf import self_train_pf
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import random
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def logging_set(mode):
    if not os.path.exists('./logging'):
        os.makedirs('./logging')
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    filemode = 'w',
                    filename='./logging/{}.log'.format(mode),
                    level = logging.INFO)
    return logging

if __name__=="__main__":
    opt = opts.parse_opt()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_modes = ['ce', 'self']

    for train_mode in train_modes:
        if train_mode == 'ce':
                opt.self_batch_size = opt.self_train_num * opt.ce_batch_size
                opt.ce_save_path = opt.ce_save_path_root

                setup_seed(opt.seed)
                train(opt, logging=logging_set('train'))
        else:
                opt.self_batch_size = opt.self_train_num * opt.ce_batch_size
                for ts_iteration in range(opt.ts_iterations):

                    opt.ce_save_path = opt.ce_save_path_root
                    opt.self_save_path = opt.self_save_path_root

                    if ts_iteration == 0:
                        # restore from the checkpoint trained with labeled data
                        opt.teacher_restore_path = os.path.join(opt.ce_save_path, opt.checkpoint_mode)
                        print(
                            'ts_training {}, load model from the {}'.format(ts_iteration, opt.teacher_restore_path))
                    else:
                        # restore from the checkpoint trained with unlabeled data
                        opt.teacher_restore_path = os.path.join(opt.self_save_path,
                                                                'model_best_{}.pth'.format(ts_iteration - 1))
                        print(
                            'ts_training {}, load model from the {}'.format(ts_iteration, opt.teacher_restore_path))
                    setup_seed(opt.seed)
                    opt.ts_iteration = ts_iteration
                    self_train_pf(opt, logging=logging_set(
                        'self_train_num_{}'.format(opt.self_train_num)))


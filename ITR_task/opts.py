import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--text_file_path', type=str, default='scripts/split_data/train_file.json',
                    help='path to the json file containing the dataset')
    parser.add_argument('--comment_file', type=str, default='data/retrieval_results_text_comment_top20.json',
                        help='path to the json file containing the comments')
    parser.add_argument('--img_feature_path', type=str, default='/home/xcp/code/EMNLP2022_final_code/data/img_text_rel/img_features',
                    help='path to the directory containing the preprocessed img fc feats')
    parser.add_argument('--self_img_feature_path', type=str,
                        default='/home/xcp/img_text_rel/extract_features_resnet152_3',
                        help='path to the directory containing the preprocessed retrieved self img fc feats')

    parser.add_argument('--save_path', type=str,
                        default='./save',
                        help='path to save the model')
    # Model settings
    parser.add_argument('--bert_hidden_size', type=int, default=768,
                    help='the hidden size of BERTweet')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                        help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_hid_size', type=int, default=300,
                        help='the attention hidden size')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='the hidden size')
    parser.add_argument('--bert_layer_num', type=int, default=6,
                    help='number of layers in the BERT')
    parser.add_argument('--trg_class', type=int, default=4,
                        help='num of target class')
    parser.add_argument('--loss_w', type=float, default=1.0,
                        help='the loss weight to balance the KL loss and CE loss')


    # Optimization: General
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='the ratio to warm up the training of the BERT')
    parser.add_argument('--seed', type=int, default=38,
                        help='random seed')
    parser.add_argument('--ce_train_batch_size', type=int, default=8,
                    help='minibatch size for CE')
    parser.add_argument('--self_train_batch_size', type=int, default=24,
                        help='minibatch size for ST')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='learning rate')
    parser.add_argument('--com_num_use', type=int, default=5,
                        help='the num of comments used in the training')
    parser.add_argument('--self_train_nums', type=int, default=3,
                        help='the num of retrieved imgs used in the training')
    parser.add_argument('--ce_save_path_root', type=str, default='save/ce_save',
                        help='path to save the ce model')
    parser.add_argument('--self_save_path_root', type=str, default='save/self_save_self_num_{}',
                        help='path to save the ST model')
    parser.add_argument('--use_com', type=bool, default=True,
                        help='use the comments or not')
    parser.add_argument('--checkpoint_mode', type=str, default='model_best.pth',
                        help='use the comments or not')
    parser.add_argument('--early_stopping_tolerance', type=int, default=2,
                        help='the epoch for early stopping')
    parser.add_argument('--ts_iterations', type=int, default=1,
                        help='the iterations for the teacher-student architecture')



    parser.add_argument('--max_epochs', type=int, default=10,
                    help='number of epochs')
    parser.add_argument('--first_mode', type=str, default='img_text_caption',
                        help='choose the first mode from [img_text, caption_text, img_text_caption]')
    parser.add_argument('--second_mode', type=str, default='multihead_text',
                        help='choose the second mode from [concat, attention, attention_text, multihead, multihead_text, co_att')

    args = parser.parse_args()



    return args
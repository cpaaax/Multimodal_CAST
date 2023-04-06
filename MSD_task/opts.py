import argparse
def parse_opt():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--text_comment_file', type=str,
                        default='./data/retrieval/retrieval_results_text_comment_top20.json')
    parser.add_argument('--img_feature_path', type=str,
                        default='/home/xcp/code/EMNLP2022_final_code/data/sarcasm/img_features')
    parser.add_argument('--img_feature_self_train_path', type=str,
                        default='/home/xcp/code/EMNLP2022_final_code/data/sarcasm/extract_features_resnet152_3')
    # parser.add_argument('--img_feature_path', type=str,
    #                     default='/home/sdb_pro/EMNLP2022_self-training_code/training_features/sarcasm/img_features')
    # parser.add_argument('--img_feature_self_train_path', type=str,
    #                     default='/home/sdb_pro/EMNLP2022_self-training_code/training_features/sarcasm/extract_features_resnet152_3')

    parser.add_argument('--ce_save_path_root', type=str, default='./save/ce_save')
    parser.add_argument('--self_save_path_root', type=str, default='./save/selftrain_save')
    parser.add_argument('--TEXT_LENGTH', type=int, default=30, help='the length of input text')


    # Model settings
    parser.add_argument('--bert_hidden_size', type=int, default=768,
                        help='the hidden size of BERTweet')
    parser.add_argument('--hidden_size', type=int, default=200,
                        help='the hidden size')
    parser.add_argument('--bert_layer_num', type=int, default=9,
                        help='number of layers in the BERT')
    parser.add_argument('--tgt_class', type=int, default=2,
                        help='num of target class')
    parser.add_argument('--loss_w', type=float, default=1.0,
                        help='the loss weight to balance the KL loss and CE loss')

    # Optimization: General
    parser.add_argument('--com_num_use', type=int, default=5,
                        help='the num of comments used in the training')
    parser.add_argument('--data_shuffle', type=bool, default=True,
                        help='shuffle the data')
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='the ratio to warm up the training of the BERT')
    parser.add_argument('--seed', type=int, default=26,
                        help='random seed')
    parser.add_argument('--ce_batch_size', type=int, default=16, help='minibatch size for CE')
    parser.add_argument('--self_batch_size', type=int, default=80, help='minibatch size for ST')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--use_com', type=bool, default=True, help='use the comments or not')
    parser.add_argument('--max_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--checkpoint_mode', type=str, default='model_best.pth')
    parser.add_argument('--dropout', type=float, default=0.0, help="dropout rate")
    parser.add_argument('--early_stopping_tolerance', type=int, default=1,
                        help='the epoch for early stopping')
    parser.add_argument('--ts_iterations', type=int, default=1,
                        help='the iterations for the teacher-student architecture')
    parser.add_argument('--self_train_num', type=int, default=3,
                        help='the num of retrieved imgs used in the training')

    args = parser.parse_args()
    return args